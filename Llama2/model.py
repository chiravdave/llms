from typing import Optional
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from utils import rotatory_freqs, apply_rotatory_emb, repeat_kv, RMSNorm


@dataclass
class ModelArgs:
	dim: int = 128
	n_layers: int = 2
	n_heads: int = 8
	n_kv_heads: Optional[int] = 2
	# This will be set using the tokenizer.
	vocab_size: int = -1
	# Making SwiGLU hidden layer size multiple of large power of 2.
	multiplier_of: int = 32
	ff_dim_multiplier: Optional[float] = None
	norm_eps: float = 1e-5
	device: str = 'cuda'
	use_flash_attn: bool = True
	
	# Needed for kv cache
	max_batch_size: int = 32
	max_seq_len: int = 512 


class SelfAttention(nn.Module):
	def __init__(self, args: ModelArgs):
		super(SelfAttention, self).__init__()
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

		# KV Cache.
		self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim)).to(args.device)
		self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim)).to(args.device)

	def forward(self, x: torch.Tensor, start_pos: int, freqs: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
		batch_size, seq_len, _ = x.shape
		
		# Calculating Q, K & V vectors.
		# (batch_size, seq_len, dim) -> (batch_size, seq_len, n_q_heads * head_dim)
		xq = self.wq(x)
		# (batch_size, seq_len, dim) -> (batch_size, seq_len, n_kv_heads * head_dim)
		xk, xv = self.wk(x), self.wv(x)

		# Generating our Q, K & V vector heads.
		# (batch_size, seq_len, n_q_heads * head_dim) -> (batch_size, seq_len, n_q_heads, head_dim)
		xq = xq.view(batch_size, seq_len, self.n_q_heads, self.head_dim)
		# (batch_size, seq_len, n_kv_heads * head_dim) -> (batch_size, seq_len, n_kv_heads, head_dim)
		xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
		# (batch_size, seq_len, n_kv_heads * head_dim) -> (batch_size, seq_len, n_kv_heads, head_dim)
		xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

		# Applying rotatory embeddings to Q & K vectors.
		xq, xk = apply_rotatory_emb(xq, xk, freqs, x.device)

		if seq_len > 1:
			# During training we don't need any cache.
			keys, values = xk, xv
		else:
			# Updating cache with the current token's calculated k & v vector.
			self.cache_k[: batch_size, start_pos : start_pos + seq_len] = xk
			self.cache_v[: batch_size, start_pos : start_pos + seq_len] = xv
			# Fetching cached k & v vectors.
			keys = self.cache_k[: batch_size, : start_pos + seq_len]
			values = self.cache_v[: batch_size, : start_pos + seq_len]

		# For every group, copying same k & v vectors for each q belonging to that group.
		# (batch_size, seq_len_kv, n_kv_heads, head_dim) -> (batch_size, seq_len_kv, n_kv_heads * groups, head_dim)
		keys, values = repeat_kv(keys, values, self.groups)

		# (batch_size, seq_len, n_q_heads, head_dim) -> (batch_size, n_q_heads, seq_len, head_dim)
		xq = xq.transpose(1, 2)
		# (batch_size, seq_len_kv, n_q_heads, head_dim) -> (batch_size, n_q_heads, seq_len_kv, head_dim)
		keys = keys.transpose(1, 2)
		# (batch_size, seq_len_kv, n_q_heads, head_dim) -> (batch_size, n_q_heads, seq_len_kv, head_dim)
		values = values.transpose(1, 2)

		attention = None
		if self.use_flash_attn:
			if mask is not None:
				attention = F.scaled_dot_product_attention(xq, keys, values, attn_mask=mask)
			else:
				attention = F.scaled_dot_product_attention(xq, keys, values, is_causal=False)
		else:
			# Calculating attention scores.
			# (_, _, seq_len, head_dim) X (_, _, seq_len_kv, head_dim) -> (batch_size, n_q_heads, seq_len, seq_len_kv)
			scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
			if mask is not None:
				scores += mask
			# (batch_size, n_q_heads, seq_len, seq_len_kv)
			scores = F.softmax(scores, dim=-1)
			# (_, _, seq_len, seq_len_kv) X (_, _, seq_len_kv, head_dim) -> (batch_size, n_q_heads, seq_len, head_dim)
			attention = torch.matmul(scores, values)
		
		attention = attention.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

		return self.wo(attention)


class FeedForward(nn.Module):
	def __init__(self, args: ModelArgs):
		super(FeedForward, self).__init__()
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
		self.ffn = FeedForward(args)
		self.attention_norm = RMSNorm(self.dim, args.norm_eps)
		self.ffn_norm = RMSNorm(self.dim, args.norm_eps)

	def forward(self, x: torch.Tensor, start_pos: int, freqs: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
		out_attention = x + self.attention(self.attention_norm(x), start_pos, freqs, mask)
		out_ff = self.ffn(self.ffn_norm(out_attention))

		return out_attention + out_ff


class Llama2(nn.Module):
	def __init__(self, args: ModelArgs):
		super(Llama2, self).__init__()
		assert args.vocab_size > 0, "Vocab size cannot be empty"
		self.args = args
		self.token_embeddings = nn.Embedding(args.vocab_size, args.dim)
		# Note that args.max_seq_len is multiplied by 2 because the token limit for the Llama 2 generation of models is 
		# 4096. Adding this multiplier instead of using 4096 directly allows for dynamism of token lengths while training 
		# or fine-tuning.
		self.freqs = rotatory_freqs(args.max_seq_len * 2, args.dim // args.n_heads, args.device)
		self.layers = nn.ModuleList()

		for _ in range(self.args.n_layers):
			self.layers.append(TransformerBlock(args))

		self.rms_norm = RMSNorm(args.dim, args.norm_eps)
		self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

	def forward(self, tokens: torch.Tensor, start_pos: int) -> torch.Tensor:
		batch_size, seq_len = tokens.shape
		# (batch_size, seq_len) -> (batch_size, seq_len, dim)
		h = self.token_embeddings(tokens)
		freqs = self.freqs[start_pos: start_pos + seq_len]

		# Generating mask.
		mask = None
		if seq_len > 1:
			# This mask is created during training where we will pass the entire sequence at once and hence the only
			# valid start_pos will be 0. For inference we will run one token at a time.
			assert start_pos == 0, "This masking will work correctly during training of entire sequence at once."
			mask = torch.full((seq_len, seq_len), float("-inf"), device=tokens.device)
			mask = torch.triu(mask, diagonal=1)


		for layer in self.layers:
			h = layer(h, start_pos, freqs, mask)

		h = self.rms_norm(h)
		
		return self.output(h)

	def save_ckpt(self, epoch, optimizer, ckpt_path) -> None:
		torch.save({
			'epoch': epoch,
			'model_state_dict': self.state_dict(),
			'optimizer_state_dict': optimizer.state_dict()
		}, ckpt_path)

	def load_ckpt(self, optimizer, ckpt_path) -> None:
		checkpoint = torch.load(ckpt_path)
		self.load_state_dict(checkpoint['model_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

	def load_weights(self, ckpt_path) -> None:
		checkpoint = torch.load(ckpt_path)
		self.load_state_dict(checkpoint['model_state_dict'])
		self.eval()
