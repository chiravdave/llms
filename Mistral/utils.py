from typing import Tuple

import torch
from torch import nn


def rotatory_freqs(seq_len: int, head_dim: int, device: str, theta: float = 10000.0) -> torch.Tensor:
	assert head_dim % 2 == 0, "Feature dimensions should be divisible by 2"
	freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2) / head_dim))
	t = torch.arange(seq_len)
	# Polar operation needs full precision.
	freqs = torch.outer(t, freqs).to(torch.float32)
	return torch.polar(torch.ones_like(freqs), freqs).to(device)


def apply_rotatory_emb(q: torch.Tensor, k: torch.Tensor, freqs: torch.Tensor, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
	q_ = torch.view_as_complex(q.float().reshape(*q.shape[:-1], -1, 2))
	k_ = torch.view_as_complex(k.float().reshape(*k.shape[:-1], -1, 2))
	# Reshape the freqs tensor to match the shape of the input tensor. So we 
	# need to add the batch dimension and the head dimension.
	# (seq_Len, head_dim/2) --> (1, seq_Len, 1, head_dim/2).
	freqs = freqs.unsqueeze(0).unsqueeze(2)
	q_rotated = torch.view_as_real(q_ * freqs).flatten(3).to(torch.get_default_dtype())
	k_rotated = torch.view_as_real(k_ * freqs).flatten(3).to(torch.get_default_dtype())

	return q_rotated.to(device), k_rotated.to(device)

def repeat_kv(k: torch.Tensor, v: torch.Tensor, groups: int) -> Tuple[torch.Tensor, torch.Tensor]:
	batch_size, seq_len, n_kv_heads, head_dim = k.shape
	if groups == 1:
		return k, v

	# (batch_size, seq_len, n_kv_heads, head_dim) -> (batch_size, seq_len, n_kv_heads, 1, head_dim) 
	repeat_k, repeat_v = k[:, :, :, None, :], v[:, :, :, None, :]
	# This operation will replicate the same k & v vectors for individual groups.
	repeat_k = repeat_k.expand(batch_size, seq_len, n_kv_heads, groups, head_dim)
	repeat_v = repeat_v.expand(batch_size, seq_len, n_kv_heads, groups, head_dim)
	# (batch_size, seq_len, n_kv_heads, groups, head_dim) -> (batch_size, seq_len, n_kv_heads * groups, head_dim)
	repeat_k = repeat_k.reshape(batch_size, seq_len, n_kv_heads * groups, head_dim)
	repeat_v = repeat_v.reshape(batch_size, seq_len, n_kv_heads * groups, head_dim)

	return repeat_k, repeat_v


class RMSNorm(nn.Module):
	def __init__(self, dim: int, eps: float = 1e-6):
		super(RMSNorm, self).__init__()
		self.eps = eps
		self.gamma = nn.Parameter(torch.ones(dim))

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
		return x * rms * self.gamma
