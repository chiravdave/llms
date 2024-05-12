from typing import List, Tuple
from os import path
from random import shuffle

import torch
from torch.nn import functional as F
from torch.nn import utils

from transformers import AutoTokenizer

from model import ModelArgs, Transformer


LLAMA2_MODEL = "meta-llama/Llama-2-7b-hf"
TEXT_FILE = "sample_data/input.txt"
EPOCH = 30
BATCH_SIZE = 32
GRADIENT_CLIP = 1.0
MAX_SEQ_LEN = 512
CKPT_PATH = "sample_data/model.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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
	if DEVICE == "cuda":
		# Training can get destabilize when setting a lower precision.
		torch.set_default_dtype(torch.float32)
	
	sentences = extract_training_sentences()
	tokenizer = AutoTokenizer.from_pretrained(LLAMA2_MODEL)
	extra_vocab = tokenizer.add_special_tokens({'pad_token': '[PAD]'})
	vocab_size = tokenizer.vocab_size + extra_vocab
	model_args = ModelArgs(vocab_size=vocab_size, device=DEVICE, 
		max_batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN)
	model = Transformer(model_args).to(DEVICE)
	# Following Llama paper.
	optimizer = torch.optim.AdamW(model.parameters(), betas=(0.9, 0.95), eps=1e-5, weight_decay=0.1)
	
	# Loading previous checkpoint if present.
	if path.exists(CKPT_PATH):
		print("Model was loaded from the latest checkpoint")
		model.load_ckpt(optimizer, CKPT_PATH)

	# Training loop starts from here.
	min_loss = float("inf")
	for epoch in range(1, EPOCH + 1):
		running_loss, batches = 0.0, 0
		for start in range(0, len(sentences), BATCH_SIZE):
			batches += 1
			batch = sentences[start : start + BATCH_SIZE]
			input_token_ids, target_tokens_ids = create_ip_op_pairs(batch, tokenizer)
			predictions = model(input_token_ids, 0)
			# Clearing gradients from last run.
			optimizer.zero_grad()
			loss = F.cross_entropy(predictions.view(-1, vocab_size), target_tokens_ids.view(-1), ignore_index=tokenizer.pad_token_id)
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
			model.save_ckpt(epoch, optimizer, CKPT_PATH)

	model.save_ckpt(epoch, optimizer, "sample_data/final_model.pt")


if __name__ == "__main__":
	train()
