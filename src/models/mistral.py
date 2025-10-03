from typing import Optional
import math

import torch
from torch import nn
from torch.nn import functional as F

from src.utils import apply_rotary_emb, ModelArgs, rotary_freqs, repeat_kv, RMSNorm


class SelfAttention(nn.Module):
	def __init__(self, args: ModelArgs):
		super(SelfAttention, self).__init__()
		self.device = args.device
		self.n_q_heads = args.n_heads
		self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
		# Groups needed in Group Based Attention mechanism.
		self.groups = self.n_q_heads // self.n_kv_heads
		# Feature size for each head.
		self.head_dim = args.dim // args.n_heads
		self.use_flash_attn = args.use_flash_attn

		# Various weight matrices required.
		self.wq = nn.Linear(args.dim, self.n_q_heads * self.head_dim, bias=False)
		self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
		self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
		self.wo = nn.Linear(args.dim, args.dim, bias=False)

	def reset_cache(self) -> None:
		self.start_pos = 0
		self.cache_k.zero_()
		self.cache_v.zero_()

	def init_cache(self, max_batch_size: int, max_seq_len: int) -> None:
		""" 
		Lazy kv cache initialization. Only created during inference.
		"""
		self.start_pos = 0
		# Rolling buffer KV Cache.
		self.cache_k = torch.zeros((max_batch_size, max_seq_len, self.n_kv_heads, self.head_dim)).to(self.device)
		self.cache_v = torch.zeros((max_batch_size, max_seq_len, self.n_kv_heads, self.head_dim)).to(self.device)

	def forward(self, x: torch.Tensor, freqs: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
		batch_size, seq_len, _ = x.shape
		
		# Calculating Q, K & V vectors.
		# (batch_size, seq_len, dim) -> (batch_size, seq_len, n_q_heads * head_dim)
		xq = self.wq(x)
		# (batch_size, seq_len, dim) -> (batch_size, seq_len, n_kv_heads * head_dim)
		xk, xv = self.wk(x), self.wv(x)

		# Changing view of our Q, K & V vectors.
		# (batch_size, seq_len, n_q_heads * head_dim) -> (batch_size, seq_len, n_q_heads, head_dim)
		xq = xq.view(batch_size, seq_len, self.n_q_heads, self.head_dim)
		# (batch_size, seq_len, n_kv_heads * head_dim) -> (batch_size, seq_len, n_kv_heads, head_dim)
		xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
		# (batch_size, seq_len, n_kv_heads * head_dim) -> (batch_size, seq_len, n_kv_heads, head_dim)
		xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

		# Applying rotatory embeddings to Q & K vectors.
		xq, xk = apply_rotary_emb(xq, xk, freqs)

		if self.training:
			# During training we don't need any cache.
			keys, values = xk, xv
		else:
			# Updating cache with the current token's calculated k & v vector.
			self.cache_k[: batch_size, self.start_pos : self.start_pos + seq_len] = xk
			self.cache_v[: batch_size, self.start_pos : self.start_pos + seq_len] = xv
			# Fetching cached k & v vectors.
			keys = self.cache_k[: batch_size, : self.start_pos + seq_len]
			values = self.cache_v[: batch_size, : self.start_pos + seq_len]
			# Updating start position by moving it with seq_len steps.
			self.start_pos += seq_len

		# For every group, copying same k & v vectors for each q belonging to that group.
		# (batch_size, seq_len, n_kv_heads, head_dim) -> (batch_size, seq_len, n_q_heads, head_dim)
		keys, values = repeat_kv(keys, values, self.groups)

		# (batch_size, seq_len, n_q_heads, head_dim) -> (batch_size, n_q_heads, seq_len, head_dim)
		xq = xq.transpose(1, 2)
		# (batch_size, seq_len, n_q_heads, head_dim) -> (batch_size, n_q_heads, seq_len, head_dim)
		keys = keys.transpose(1, 2)
		# (batch_size, seq_len, n_q_heads, head_dim) -> (batch_size, n_q_heads, seq_len, head_dim)
		values = values.transpose(1, 2)

		attention = None
		if self.use_flash_attn:
			attention = F.scaled_dot_product_attention(xq, keys, values, is_causal=True)
		else:
			# Calculating attention scores.
			# (_, _, seq_len, head_dim) X (_, _, seq_len_kv, head_dim) -> (batch_size, n_q_heads, seq_len, seq_len_kv)
			scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
			scores += mask
			# (batch_size, n_q_heads, seq_len, seq_len_kv)
			scores = F.softmax(scores, dim=-1)
			# (_, _, seq_len, seq_len_kv) X (_, _, seq_len_kv, head_dim) -> (batch_size, n_q_heads, seq_len, head_dim)
			attention = torch.matmul(scores, values)
		
		# (batch_size, n_q_heads, seq_len, head_dim) -> (batch_size, seq_len, dim)
		attention = attention.transpose(1, 2).reshape(batch_size, seq_len, -1)

		return self.wo(attention)

class MOELayer(nn.Module):
	def __init__(self, args: ModelArgs):
		super(MOELayer, self).__init__()
		self.num_experts = args.num_experts
		self.topk = args.topk
		self.experts = nn.ModuleList([FeedForward(args) for _ in range(args.num_experts)])
		self.gate = nn.Linear(args.dim, args.num_experts, bias=False)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		batch_size, seq_len, dim = x.shape
		# (batch_size * seq_len, dim)
		x_flat = x.view(-1, dim)

		# 1. Compute gating logits, top-k expert idxs & top-k weights.
		# (batch_size, seq_len, topk)
		gate_logits = self.gate(x)
		# (batch_size, seq_len, topk), (batch_size, seq_len, topk)
		topk_weights, topk_indices = torch.topk(gate_logits, self.topk, dim=-1)
		topk_weights = F.softmax(topk_weights, dim=-1)
		
		# 2. Flatten matrices for efficient processing. A contiguous array is created so that GPU kernels can operate 
  		# faster by leveraging memory coalescing. 
		# (batch_size * seq_len * topk, )
		topk_weights = topk_weights.view(batch_size * seq_len * self.topk)
		# (batch_size * seq_len * topk, )
		topk_indices = topk_indices.view(batch_size * seq_len * self.topk)

		# 3. Create a flat token indices array where each token is repeated top-k times.
		# (batch_size * seq_len, ) -> (batch_size * seq_len, 1) -> (batch_size * seq_len, top-k) -> 
  		# (batch_size * seq_len * top-k, ).  
		token_indices = torch.arange(
			batch_size * seq_len, device=x.device).unsqueeze(-1).expand(-1, self.topk).reshape(-1)

		# 4. Prepare expert inputs & outputs for all the token indices from previous step.
		# (batch_size * seq_len * top-k, dim)
		expert_inputs = x_flat[token_indices]
		# (batch_size * seq_len * top-k, dim)
		expert_outputs = torch.zeros_like(expert_inputs)

		# 5. Run each expert with inputs belonging to it, one at a time.
		for expert_id in range(self.num_experts):
			# Create a boolean mask with shape (batch_size * seq_len * topk) where elements are True if they match the 
			# given expert_id, otherwise False.
			mask = topk_indices == expert_id
			# Skip the expert if it is not contributing to any token.
			if not mask.any():
				continue
			# (tokens for the expert, dim)
			inputs_for_expert = expert_inputs[mask]
			# (tokens for the expert, dim)
			outputs_from_expert = self.experts[expert_id](inputs_for_expert)
			# Setting the expert's decision for the selected tokens.
			expert_outputs[mask] = outputs_from_expert.to(expert_outputs.dtype)

		# 6. Weight expert outputs with gating weights. Unsqueeze allows for safe broadcasting. Each element of 'dim' 
		# dimension in a row gets updated with the corresponding gate weight. 
		weighted_expert_outputs = expert_outputs * topk_weights.unsqueeze(-1)

		# 7. Now scatter the weighted outputs back to original token positions.
		# (batch_size * seq_len, dim)
		final_output = torch.zeros_like(x_flat)
		# Accumulates weighted expert outputs into final_output at positions specified by token_indices.
		final_output = final_output.index_add_(0, token_indices, weighted_expert_outputs)

		# 8. Reshape back to (batch_size, seq_len, dim)
		return final_output.view(batch_size, seq_len, dim)

class FeedForward(nn.Module):
	def __init__(self, args: ModelArgs):
		super(FeedForward, self).__init__()
		# Start with 4x expansion (standard transformer practice)
		hidden_dim = args.dim * 4
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
		self.attention_norm = RMSNorm(self.dim, args.norm_eps)
		self.ffn_norm = RMSNorm(self.dim, args.norm_eps)

		if args.num_experts > 1:
			print("Using MOELayer")
			self.ffn = MOELayer(args)
		else:
			self.ffn = FeedForward(args)

	def forward(self, x: torch.Tensor, freqs: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
		out_attention = x + self.attention(self.attention_norm(x), freqs, mask)
		out_ff = self.ffn(self.ffn_norm(out_attention))

		return out_attention + out_ff

class Mistral(nn.Module):
	def __init__(self, args: ModelArgs):
		super(Mistral, self).__init__()
		assert args.vocab_size > 0, "Vocab size cannot be empty"
		assert args.dim % args.n_heads == 0, "Hidden dimension should be multiple of total heads"
		assert args.n_heads % args.n_kv_heads == 0, "Total heads should be multiple of total kv heads"
		self.args = args
		self.token_embeddings = nn.Embedding(args.vocab_size, args.dim)
		# Note that args.max_seq_len is multiplied by 2 because the token limit for the Llama 2 generation of models is 
		# 4096. Adding this multiplier instead of using 4096 directly allows for dynamism of token lengths while 
		# training or fine-tuning.
		self.freqs = rotary_freqs(args.max_seq_len * 2, args.dim // args.n_heads, args.device)
		self.layers = nn.ModuleList()

		for _ in range(self.args.n_layers):
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
		if not self.args.use_flash_attn:
			mask = torch.full((seq_len, seq_len), float("-inf"), device=batch.device)
			mask = torch.triu(mask, diagonal=1)

		return self.forward_pass(embeddings, freqs, mask)

	def forward_pass(self, batch: torch.Tensor, freqs: torch.Tensor, mask: Optional[torch.Tensor] = None
	) -> torch.Tensor:
		for layer in self.layers:
			batch = layer(batch, freqs, mask)

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

	def eval(self) -> None:
		"""
		Set the model in eval mode. Will also prepare kv cache for faster inference.
		"""
		super().eval()
		for layer in self.layers:
			layer.attention.init_cache(self.args.max_batch_size, self.args.max_seq_len)

	def generate(self, batch: torch.Tensor, eos_token_id: int, temperature: float = 1.0) -> torch.Tensor:
		"""
		Generates text autoregressively with KV caching. In the first stage it runs prefills - calculating kv values 
		for the input prefix. From first stage onwards, it runs decodings - generating one token at a time. 
		"""
		batch_size, seq_len = batch.shape
		assert batch_size == 1, "Currently supported batch size is 1"
		assert temperature >= 0.0 and temperature <= 1.0, "Valid range of temperature is: 0-1"

		if self.training:
			print("Model is not in eval mode. Setting the model to eval mode.")
			self.eval()
		
		# Reset previous cache.
		for layer in self.layers:
			layer.attention.reset_cache()

		# All but last token. This is equivalent to prefill stage.
		embeddings = self.token_embeddings(batch)
		start_pos, output_tokens = 0, list()
		while seq_len < self.args.max_seq_len:
			logits = self.forward_pass(embeddings, self.freqs[start_pos:seq_len])[:, -1, :]
			if temperature == 0.0:
				probs = F.softmax(logits, dim=-1)
				next_batch = torch.argmax(probs, dim=-1, keepdim=True)
			else:
				probs = F.softmax(logits / temperature, dim=-1)
				next_batch = torch.multinomial(probs, num_samples=1)

			output_tokens.append(next_batch)
			start_pos = seq_len
			seq_len += 1
			if (next_batch == eos_token_id).all() or seq_len == self.args.max_seq_len:
				break

			embeddings = self.token_embeddings(next_batch)

		return torch.cat(output_tokens, dim=1)