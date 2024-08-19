from typing import Any, Dict
from os import path

import torch
from torch.nn import functional as F
from torch.nn import utils
from torch.utils.data import DataLoader


class Trainer:
	def __init__(self, model, tokenizer, optimizer, epoch: int, train_dataloader: DataLoader, model_save_dir: str, 
		checkpoint: str, kwargs: Dict[str, Any]):
		self.model = model
		self.tokenizer = tokenizer
		self.optimizer = optimizer
		self.epoch = epoch
		self.train_dataloader = train_dataloader
		self.model_save_dir = model_save_dir
		self.ckpt = checkpoint
		self.kwargs = kwargs

	def train(self):
		device = self.kwargs.get('DEVICE', "cpu")
		device_type = self.kwargs.get('DEVICE_TYPE', "cpu")
		use_mix_precision = self.kwargs.get('USE_MIX_PRECISION', True)
		gradient_clip = self.kwargs.get('GRADIENT_CLIP', 1.0)
		min_loss = float("inf")
		for epoch in range(1, self.epoch + 1):
			running_loss = 0.0
			for input_token_ids, target_token_ids in self.train_dataloader:
				input_token_ids, target_token_ids = input_token_ids.to(device), target_token_ids.to(device)
				# Clearing gradients from last run.
				self.optimizer.zero_grad()
				with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=use_mix_precision):
					predictions = self.model(input_token_ids, 0)
					loss = F.cross_entropy(
						predictions.view(-1, len(self.tokenizer)), target_token_ids.view(-1), 
						ignore_index=self.tokenizer.pad_token_id
					)
				# Calculte loss backwards and update model weights.
				loss.backward()
				# Clipping Gradients.
				utils.clip_grad_norm_(self.model.parameters(), max_norm=gradient_clip)
				self.optimizer.step()
				running_loss += loss.item()

			epoch_loss = running_loss / len(self.train_dataloader)
			print(f"Loss after {epoch} epoch: {epoch_loss}")
			if min_loss > epoch_loss:
				min_loss = epoch_loss
				self.model.save_ckpt(epoch, self.optimizer, path.join(self.model_save_dir, self.ckpt))
