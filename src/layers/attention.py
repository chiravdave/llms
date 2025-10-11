from typing import Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import apply_rotary_emb, ModelArgs, repeat_kv


class SelfAttention(nn.Module):
	"""
	Multi-Head Self Attention.
	"""
	def __init__(self, args: ModelArgs):
		super(SelfAttention, self).__init__()
		self.device = args.device
		self.n_q_heads = args.n_heads
		# Feature size for each head.
		self.head_dim = args.dim // args.n_heads
		self.n_kv_heads = args.n_heads if args.n_kv_heads == args.n_heads else args.n_kv_heads
		# Groups needed in Group Based Attention mechanism.
		self.groups = self.n_q_heads // self.n_kv_heads
		self.use_flash_attn = args.use_flash_attn

		# Various weight matrices required.
		self.W_Q = nn.Linear(args.dim, self.n_q_heads * self.head_dim, bias=False)
		self.W_K = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
		self.W_V = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
		self.W_O = nn.Linear(args.dim, args.dim, bias=False)

	def reset_cache(self) -> None:
		self.start_pos = 0
		self.cache_k.zero_()
		self.cache_v.zero_()

	def init_cache(self, max_batch_size: int, max_seq_len: int) -> None:
		""" 
		Lazy kv cache initialization. Only created during inference.
		"""
		self.start_pos = 0
		self.cache_k = torch.zeros((max_batch_size, max_seq_len, self.n_kv_heads, self.head_dim)).to(self.device)
		self.cache_v = torch.zeros((max_batch_size, max_seq_len, self.n_kv_heads, self.head_dim)).to(self.device)

	def forward(self, x: torch.Tensor, freqs: torch.Tensor, mask: Optional[torch.Tensor] = None, use_cache=False
	) -> torch.Tensor:
		batch_size, seq_len, _ = x.shape
		
		# Calculating Q, K & V vectors.
		# (_, seq_len, dim) @ (dim, n_heads * head_dim) -> (_, seq_len, n_q_heads * head_dim)
		Q = self.W_Q(x)
		# (_, seq_len, dim) @ (dim, n_kv_heads * head_dim) -> (_, seq_len, n_kv_heads * head_dim)
		K, V = self.W_K(x), self.W_V(x)

		# Changing view of our Q, K & V vectors.
		# (_, seq_len, n_q_heads * head_dim) -> (_, seq_len, n_q_heads, head_dim)
		Q = Q.view(batch_size, seq_len, self.n_q_heads, self.head_dim)
		# (_, seq_len, n_kv_heads * head_dim) -> (_, seq_len, n_kv_heads, head_dim)
		K = K.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
		# (_, seq_len, n_kv_heads * head_dim) -> (_, seq_len, n_kv_heads, head_dim)
		V = V.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

		# Applying rotatory embeddings to Q & K vectors.
		Q, K = apply_rotary_emb(Q, K, freqs)

		# Cache calculation. During training we don't need any cache.
		if not self.training and use_cache:
			# Updating cache with the current token's calculated k & v vector.
			self.cache_k[: batch_size, self.start_pos : self.start_pos + seq_len] = K
			self.cache_v[: batch_size, self.start_pos : self.start_pos + seq_len] = V
			# Fetching cached k & v vectors.
			K = self.cache_k[: batch_size, : self.start_pos + seq_len]
			V = self.cache_v[: batch_size, : self.start_pos + seq_len]
			# Updating start position by moving it with seq_len steps.
			self.start_pos += seq_len 

		# For every group, copying same k & v vectors for each q belonging to that group.
		# (_, seq_len, n_kv_heads, head_dim) -> (_, seq_len, n_q_heads, head_dim)
		K, V = repeat_kv(K, V, self.groups)

		# (_, seq_len, n_q_heads, head_dim) -> (_, n_q_heads, seq_len, head_dim)
		Q = Q.transpose(1, 2)
		# (_, seq_len, n_q_heads, head_dim) -> (_, n_q_heads, seq_len, head_dim)
		K = K.transpose(1, 2)
		# (_, seq_len, n_q_heads, head_dim) -> (_, n_q_heads, seq_len, head_dim)
		V = V.transpose(1, 2)

		attention = None
		if self.use_flash_attn:
			attention = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
		else:
			# Calculating attention scores.
			# (_, n_q_heads, seq_len, head_dim) @ (_, n_q_heads, head_dim, seq_len) -> (_, n_q_heads, seq_len, seq_len)
			scores = torch.matmul(Q, K.transpose(2, 3)) / math.sqrt(self.head_dim)
			scores += mask
			# (_, n_q_heads, seq_len, seq_len)
			scores = F.softmax(scores, dim=-1)
			# (_, _, seq_len, seq_len) @ (_, n_q_heads, seq_len, head_dim) -> (_, n_q_heads, seq_len, head_dim)
			attention = torch.matmul(scores, V)
		
		# (_, n_q_heads, seq_len, head_dim) -> (_, seq_len, dim)
		attention = attention.transpose(1, 2).reshape(batch_size, seq_len, -1)

		return self.W_O(attention)

class MultiHeadLatentAttention(nn.Module):
	"""
	Multi-Head Latent Attention (MLA) as introduced by DeepSeek.

	Key innovation: Compresses K and V into a low-rank latent space to reduce KV cache.
	"""
	def __init__(self, args: ModelArgs):
		super(MultiHeadLatentAttention, self).__init__()
		self.device = args.device
		self.n_heads = args.n_heads
		# Feature size for each head.
		self.head_dim = args.dim // args.n_heads
		self.latent_dim = args.latent_dim
		self.use_flash_attn = args.use_flash_attn

		# Various weight matrices required.
		self.W_Q = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
		# Compression: Project input to latent space.
		self.W_down = nn.Linear(args.dim, args.latent_dim, bias=False)
		# Decompression: Up-projections for K and V.
		self.W_K_up = nn.Linear(args.latent_dim, args.n_heads * self.head_dim, bias=False)
		self.W_V_up = nn.Linear(args.latent_dim, args.n_heads * self.head_dim, bias=False)
		self.W_O = nn.Linear(args.dim, args.dim, bias=False)
		
		# Absorption Trick: Pre-compute absorbed weights. Only useful during inferencing.
		if not self.training:
			self._init_absorbed_weights()
	
	def _init_absorbed_weights(self):
		"""
		Absorption Trick: Pre-multiply W_down with W_K_up and W_V_up. This saves two matmul operations. Ex:

		Instead of: c_kv = X @ W_down, then K = c_kv @ W_K_up
		We compute: K = X @ (W_down @ W_K_up) = X @ W_K_absorbed
		"""
		with torch.no_grad():
			# Absorbed weights: (dim, latent_dim) @ (latent_dim, n_heads * head_dim) -> (dim, n_heads * head_dim)
			self.W_K_absorbed = self.W_down.weight.T @ self.W_K_up.weight.T
			self.W_V_absorbed = self.W_down.weight.T @ self.W_V_up.weight.T

	def reset_cache(self) -> None:
		self.start_pos = 0
		self.cache_kv.zero_()

	def init_cache(self, max_batch_size: int, max_seq_len: int) -> None:
		""" 
		Lazy kv cache initialization. Only created during inference.
		"""
		self.start_pos = 0
		self.cache_kv = torch.zeros((max_batch_size, max_seq_len, self.latent_dim)).to(self.device)
	
	def forward(self, x: torch.Tensor, freqs: torch.Tensor, mask: Optional[torch.Tensor] = None, use_cache=False
		) -> torch.Tensor:
		batch_size, seq_len, _ = x.shape
		
		# Calculating Q, K & V vectors.
		# (_, seq_len, dim) @ (dim, n_heads * head_dim) -> (_, seq_len, n_heads * head_dim)
		Q = self.W_Q(x)
		# Compress to latent space. 
		# (_, seq_len, dim) @ (dim, latent_dim) -> (_, seq_len, latent_dim)
		c_kv = self.W_down(x)
		# During training we don't use absorbtion trick.
		if self.training:
			# (_, seq_len, latent_dim) @ (latent_dim, n_heads * head_dim) -> (_, seq_len, n_heads * head_dim)
			K = self.W_K_up(c_kv)
			V = self.W_V_up(c_kv)
		else:
			if use_cache:
				# Updating cache with the current token's calculated latent vector.
				self.cache_kv[: batch_size, self.start_pos : self.start_pos + seq_len] = c_kv
				# Fetching cached latent vectors.
				c_kv = self.cache_kv[: batch_size, : self.start_pos + seq_len]
				# (_, seq_len, latent_dim) @ (latent_dim, n_heads * head_dim) -> (_, seq_len, n_heads * head_dim)
				K = self.W_K_up(c_kv)
				V = self.W_V_up(c_kv)
			else:
				# Use absorbtion trick for K and V computation.
				# (_, seq_len, dim) @ (dim, n_heads * head_dim) -> (_, seq_len, n_heads * head_dim)
				K = x @ self.W_K_absorbed
				V = x @ self.W_V_absorbed

		# Changing view of our Q, K & V vectors.
		# (_, _, n_heads * head_dim) -> (_, _, n_heads, head_dim)
		Q = Q.view(batch_size, seq_len, self.n_heads, self.head_dim)
		# (_, _, n_heads * head_dim) -> (_, _, n_heads, head_dim)
		K = K.view(batch_size, seq_len, self.n_heads, self.head_dim)
		# (_, _, n_heads * head_dim) -> (_, _, n_heads, head_dim)
		V = V.view(batch_size, seq_len, self.n_heads, self.head_dim)

		# Applying rotatory embeddings to Q & K vectors.
		Q, K = apply_rotary_emb(Q, K, freqs)

		# (_, seq_len, n_heads, head_dim) -> (_, n_heads, seq_len, head_dim)
		Q = Q.transpose(1, 2)
		# (_, seq_len, n_heads, head_dim) -> (_, n_heads, seq_len, head_dim)
		K = K.transpose(1, 2)
		# (_, seq_len, n_heads, head_dim) -> (_, n_heads, seq_len, head_dim)
		V = V.transpose(1, 2)
		
		attention = None
		if self.use_flash_attn:
			attention = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
		else:
			# Calculating attention scores.
			# (_, n_heads, seq_len, head_dim) @ (_, n_heads, head_dim, seq_len) -> (_, n_heads, seq_len, seq_len)
			scores = torch.matmul(Q, K.transpose(2, 3)) / math.sqrt(self.head_dim)
			scores += mask
			# (_, n_heads, seq_len, seq_len)
			scores = F.softmax(scores, dim=-1)
			# (_, n_heads, seq_len, seq_len) @ (_, _, seq_len, head_dim) -> (_, n_heads, seq_len, head_dim)
			attention = torch.matmul(scores, V)
		
		# (_, n_heads, seq_len, head_dim) -> (_, seq_len, dim)
		attention = attention.transpose(1, 2).reshape(batch_size, seq_len, -1)

		return self.W_O(attention)