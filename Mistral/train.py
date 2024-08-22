from os import path, makedirs

import torch
from torch.utils.data import DataLoader

from transformers import AutoTokenizer

from model import ModelArgs, Mistral
from dataloader import ShakespearDataset
from trainer import Trainer
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


def train():
	cleanups()
	setups(DEVICE_TYPE)

	# Loading tokenizer.
	tokenizer = None
	if path.exists(MODEL_SAVE_DIR):
		print("Tokenizer was loaded from the latest checkpoint.")
		tokenizer = AutoTokenizer.from_pretrained(MODEL_SAVE_DIR, local_files_only=True)
	else:
		tokenizer = AutoTokenizer.from_pretrained(MISTRAL_TOKENIZER, cache_dir="models")
		tokenizer.add_special_tokens({'pad_token': '[PAD]'})
		makedirs(MODEL_SAVE_DIR)
		tokenizer.save_pretrained(MODEL_SAVE_DIR)
	
	# Loading model & optimizer.
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

	# Loading datasets
	train_dataset = ShakespearDataset(tokenizer, MAX_SEQ_LEN, TEXT_FILE)
	train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
	
	# Training begins here.
	trainer = Trainer(model, tokenizer, optimizer, EPOCH, train_dataloader, MODEL_SAVE_DIR, CKPT_FILE, 
		{'DEVICE': DEVICE, 'DEVICE_TYPE': DEVICE_TYPE, 'USE_MIX_PRECISION': USE_MIX_PRECISION, 'GRADIENT_CLIP': GRADIENT_CLIP}
	)
	trainer.train()
	
	# model.save_ckpt(epoch, optimizer, path.join(MODEL_SAVE_DIR, "final_model.pt"))
	cleanups()


if __name__ == "__main__":
	train()
