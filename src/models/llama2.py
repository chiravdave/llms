from typing import Any, Dict, Optional

import torch
from torch import nn
from torch.nn import functional as F

from src.layers.attention import SelfAttention
from src.layers.norm import RMSNorm
from src.utils import ModelArgs, rotary_freqs


class FeedForward(nn.Module):
	def __init__(self, args: ModelArgs):
		super(FeedForward, self).__init__()
		# Start with 4x expansion (standard transformer practice)
		hidden_dim = args.dim * 4
		# Apply 2/3 reduction (SwiGLU-specific)
		hidden_dim = (2 * hidden_dim) // 3
		if args.ff_dim_multiplier:
			hidden_dim = int(args.ff_dim_multiplier * hidden_dim)

		# Rounding the hidden_dim to the nearest multiple of the multiple_of parameter.
		hidden_dim = args.multiplier_of * ((hidden_dim + args.multiplier_of - 1) // args.multiplier_of)
		self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
		self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
		# Alternate path of linear transformation.
		self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		out_w1 = F.silu(self.w1(x))
		out_w3 = self.w3(x)

		return self.w2(out_w1 * out_w3)

class TransformerBlock(nn.Module):
	def __init__(self, args: ModelArgs):
		super(TransformerBlock, self).__init__()
		self.n_heads = args.n_heads
		self.dim = args.dim
		self.head_dim = self.dim // self.n_heads
		self.attention = SelfAttention(args)
		self.ffn = FeedForward(args)
		self.attention_norm = RMSNorm(self.dim, args.norm_eps)
		self.ffn_norm = RMSNorm(self.dim, args.norm_eps)

	def forward(self, x: torch.Tensor, freqs: torch.Tensor, mask: Optional[torch.Tensor] = None, use_cache: bool = False
	) -> torch.Tensor:
		out_attention = x + self.attention(self.attention_norm(x), freqs, mask, use_cache)
		out_ff = self.ffn(self.ffn_norm(out_attention))

		return out_attention + out_ff

class Llama2(nn.Module):
	def __init__(self, args: ModelArgs):
		super(Llama2, self).__init__()
		assert args.vocab_size > 0, "Vocab size cannot be empty"
		assert args.dim % args.n_heads == 0, "Hidden dimension should be multiple of total heads"
		assert args.n_heads % args.n_kv_heads == 0, "Total heads should be multiple of total kv heads"
		self.use_flash_attn = args.use_flash_attn
		self.token_embeddings = nn.Embedding(args.vocab_size, args.dim)
		# Note that args.max_seq_len is multiplied by 2 because the token limit for the Llama 2 generation of models is 
		# 4096. Adding this multiplier instead of using 4096 directly allows for dynamism of token lengths while 
		# training or fine-tuning.
		self.freqs = rotary_freqs(args.max_seq_len * 2, args.dim // args.n_heads, args.device)
		self.layers = nn.ModuleList()

		for _ in range(args.n_layers):
			self.layers.append(TransformerBlock(args))

		self.rms_norm = RMSNorm(args.dim, args.norm_eps)
		self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

	def forward(self, batch: torch.Tensor) -> torch.Tensor:
		_, seq_len = batch.shape
		# (batch_size, seq_len) -> (batch_size, seq_len, dim)
		embeddings = self.token_embeddings(batch)
		freqs = self.freqs[:seq_len]

		# Generating mask.
		mask = None
		if not self.use_flash_attn:
			mask = torch.full((seq_len, seq_len), float("-inf"), device=batch.device)
			mask = torch.triu(mask, diagonal=1)

		return self.forward_pass(embeddings, freqs, mask)

	def forward_pass(
		self, batch: torch.Tensor, freqs: torch.Tensor, mask: Optional[torch.Tensor] = None, use_cache: bool = False
	) -> torch.Tensor:
		for layer in self.layers:
			batch = layer(batch, freqs, mask, use_cache)

		batch = self.rms_norm(batch)
		return self.output(batch)

	def save_ckpt(self, epoch: int, ckpt_path: str, optimizer = None) -> None:
		details = {
			'epoch': epoch,
			'model_state_dict': self.state_dict(),
		}
		if optimizer:
			details['optimizer_state_dict'] = optimizer.state_dict()
		
		torch.save(details, ckpt_path)

	def load_ckpt(self, ckpt_path: str, optimizer = None) -> int:
		checkpoint = torch.load(ckpt_path)
		self.load_state_dict(checkpoint['model_state_dict'])
		if optimizer:
			if 'optimizer_state_dict' not in checkpoint:
				print("No optimizer states found in the checkpoint file")
			else:
				optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

		return checkpoint['epoch']

	def eval(self, max_batch_size: int, max_seq_len: int) -> None:
		"""
		Set the model in eval mode. Will also prepare kv cache for faster inference.
		"""
		super().eval()
		for layer in self.layers:
			layer.attention.init_cache(max_batch_size, max_seq_len)

	def generate(self, batch: torch.Tensor, **kwargs: Dict[str, Any]) -> torch.Tensor:
		"""
		Generates text autoregressively with KV caching. In the first stage it runs prefills - calculating kv values 
		for the input prefix. From first stage onwards, it runs decodings - generating one token at a time. 
		"""
		temperature = kwargs.get('TEMPERATURE', 1.0)
		max_seq_len = kwargs.get('MAX_SEQ_LEN', 512)
		batch_size, seq_len = batch.shape
		assert batch_size == 1, "Currently supported batch size is 1"
		assert temperature >= 0.0 and temperature <= 1.0, "Valid range of temperature is: 0-1"

		if self.training:
			print("Model is not in eval mode. Setting the model to eval mode.")
			self.eval(batch_size, max_seq_len)
		
		# Reset previous cache.
		for layer in self.layers:
			layer.attention.reset_cache()

		# All but last token. This is equivalent to prefill stage.
		embeddings = self.token_embeddings(batch)
		start_pos, output_tokens = 0, list()
		while seq_len < max_seq_len:
			logits = self.forward_pass(
				embeddings, self.freqs[start_pos:seq_len], use_cache=kwargs.get('USE_CACHE', True)
			)[:, -1, :]
			if temperature == 0.0:
				probs = F.softmax(logits, dim=-1)
				next_batch = torch.argmax(probs, dim=-1, keepdim=True)
			else:
				probs = F.softmax(logits / temperature, dim=-1)
				next_batch = torch.multinomial(probs, num_samples=1)

			output_tokens.append(next_batch)
			start_pos = seq_len
			seq_len += 1
			if (next_batch == kwargs['EOS_TOKEN_ID']).all() or seq_len == max_seq_len:
				break

			embeddings = self.token_embeddings(next_batch)

		return torch.cat(output_tokens, dim=1)