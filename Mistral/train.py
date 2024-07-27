from typing import List, Tuple
from os import path, makedirs
from random import shuffle

import torch
from torch.nn import functional as F
from torch.nn import utils

from transformers import AutoTokenizer

from model import ModelArgs, Mistral
from utils import cleanups, setups, configure_optimizer


MISTRAL_TOKENIZER = "mistralai/Mistral-7B-v0.1"
MODEL_SAVE_DIR = "models/mistral7b"
CKPT_FILE = "model.pt"
TEXT_FILE = "input.txt"
EPOCH = 100
BATCH_SIZE = 256
GRADIENT_CLIP = 1.0
MAX_SEQ_LEN = 512
USE_MIX_PRECISION = True
DEVICE_TYPE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "cuda:0"


def create_ip_op_pairs(sentences: List[str], tokenizer) -> Tuple[torch.Tensor, torch.Tensor]:
	"""
	Function to create input & output pair for text generation. Output tokens will be shifted by one position to the
	left.
	"""
	input_token_ids = tokenizer(sentences, padding="longest", truncation=True, 
		max_length=MAX_SEQ_LEN, return_tensors="pt", return_attention_mask =False)['input_ids']
	# Generating target token ids having eos_token_id and shifted one position 
	# to the left.
	output_token_ids = torch.roll(input_token_ids, -1, 1)
	output_token_ids[:, -1] = tokenizer.pad_token_id

	for ids in output_token_ids:
		first_pad_index = (ids == tokenizer.pad_token_id).nonzero(as_tuple=True)[0][0]
		ids[first_pad_index] = tokenizer.eos_token_id

	return input_token_ids.to(DEVICE), output_token_ids.to(DEVICE)


def extract_training_sentences() -> List[str]:
	"""
	This function will generate sentences from the dataset file. Every sentence will have at max 100 words.
	"""
	sentences = list()
	with open(TEXT_FILE, 'r') as f:
		total_words, words = 0, list()
		for line in f.readlines():
			for word in line.split(' '):
				total_words += 1
				words.append(word)
				if total_words == 100:
					total_words = 0
					sentences.append(' '.join(words))
					words.clear()

	return sentences


def train():
	cleanups()
	setups(DEVICE_TYPE)

	tokenizer = None
	if path.exists(MODEL_SAVE_DIR):
		print("Tokenizer was loaded from the latest checkpoint.")
		tokenizer = AutoTokenizer.from_pretrained(MODEL_SAVE_DIR, local_files_only=True)
	else:
		tokenizer = AutoTokenizer.from_pretrained(MISTRAL_TOKENIZER, cache_dir="models")
		tokenizer.add_special_tokens({'pad_token': '[PAD]'})
		makedirs(MODEL_SAVE_DIR)
		tokenizer.save_pretrained(MODEL_SAVE_DIR)
	
	vocab_size = len(tokenizer)
	model_args = ModelArgs(vocab_size=vocab_size, device=DEVICE, max_batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN)
	model = Mistral(model_args).to(DEVICE)
	optimizer = configure_optimizer(model.named_parameters(), lr_rate=6e-4, weight_decay=0.1, device_type=DEVICE_TYPE)
	
	# Loading the model from previous checkpoint if present.
	if path.exists(path.join(MODEL_SAVE_DIR, CKPT_FILE)):
		print("Model was loaded from the latest checkpoint.")
		model.load_ckpt(optimizer, path.join(MODEL_SAVE_DIR, CKPT_FILE))

	# Will compile the model once and hence all layers will run faster during forward calls.
	model = torch.compile(model)
	# Training loop starts from here.
	min_loss, sentences = float("inf"), extract_training_sentences()
	for epoch in range(1, EPOCH + 1):
		running_loss, batches = 0.0, 0
		for start in range(0, len(sentences), BATCH_SIZE):
			batches += 1
			batch = sentences[start : start + BATCH_SIZE]
			input_token_ids, target_tokens_ids = create_ip_op_pairs(batch, tokenizer)
			# Clearing gradients from last run.
			optimizer.zero_grad()
			with torch.autocast(device_type=DEVICE_TYPE, dtype=torch.bfloat16, enabled=USE_MIX_PRECISION):
				predictions = model(input_token_ids, 0)
				loss = F.cross_entropy(
					predictions.view(-1, vocab_size), target_tokens_ids.view(-1), ignore_index=tokenizer.pad_token_id
				)
			# Calculte loss backwards and update model weights.
			loss.backward()
			# Clipping Gradients.
			utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIP)
			optimizer.step()
			running_loss += loss.item()

		epoch_loss = running_loss / batches
		print(f"Loss after {epoch} epoch: {epoch_loss}")
		shuffle(sentences)
		if min_loss > epoch_loss:
			min_loss = epoch_loss
			model.save_ckpt(epoch, optimizer, path.join(MODEL_SAVE_DIR, CKPT_FILE))

	model.save_ckpt(epoch, optimizer, path.join(MODEL_SAVE_DIR, "final_model.pt"))
	cleanups()


if __name__ == "__main__":
	train()
