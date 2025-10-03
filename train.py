from typing import Tuple
from argparse import ArgumentParser
from os import environ, makedirs, path
from functools import partial

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.distributed import init_process_group, destroy_process_group
from transformers import AutoTokenizer

from src.models.llama2 import Llama2
from src.models.mistral import Mistral
from src.dataloader import ShakespearDataset, generate_ip_op_pairs
from src.trainer import Trainer
from src.utils import cleanups, configure_optimizer, ModelArgs, setups


TOKENIZER = "Qwen/Qwen3-4B"
HF_MODEL_DIR = "/root/.cache/huggingface/hub"
MODEL_SAVE_DIR = "trained-models/ddp/small-llama"
MODEL_NAME = "llama2" 							# Update model name here to run different model
CKPT_FILE = "model.pt"
TEXT_FILE = "data/input.txt"
EPOCH = 30
BATCH_SIZE = 16
GRADIENT_CLIP = 1.0
MAX_SEQ_LEN = 1024
USE_MIX_PRECISION = True
DEVICE_TYPE = "cuda" if torch.cuda.is_available() else "cpu"


def ddp_setup():
	init_process_group(backend="nccl")
	torch.cuda.set_device(int(environ["LOCAL_RANK"]))

def load_model_and_tokenizer(device: str) -> Tuple[nn.Module, AutoTokenizer, AdamW]:
	# Loading tokenizer.
	tokenizer = None
	if path.exists(MODEL_SAVE_DIR):
		print("Tokenizer was loaded from the latest checkpoint.")
		tokenizer = AutoTokenizer.from_pretrained(MODEL_SAVE_DIR, local_files_only=True)
	else:
		tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, cache_dir=HF_MODEL_DIR)
		tokenizer.add_special_tokens({'pad_token': '[PAD]'})
		makedirs(MODEL_SAVE_DIR)
		tokenizer.save_pretrained(MODEL_SAVE_DIR)

	# Loading model.
	model_args = ModelArgs(vocab_size=len(tokenizer), device=device, max_batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN)
	model = None
	if MODEL_NAME == "llama2":
			model = Llama2(model_args).to(device)
	elif MODEL_NAME == "mistral":
		model = Mistral(model_args).to(device)
	else:
		print(f"Invalid model name provided: {MODEL_NAME}")
		exit()

	optimizer = configure_optimizer(model.named_parameters(), lr_rate=6e-4, weight_decay=0.1, device_type=DEVICE_TYPE)
	# Loading the model from previous checkpoint if present.
	if path.exists(path.join(MODEL_SAVE_DIR, CKPT_FILE)):
		print("Model was loaded from the latest checkpoint.")
		model.load_ckpt(path.join(MODEL_SAVE_DIR, CKPT_FILE), optimizer)

	return model, tokenizer, optimizer

def train_ddp():
	device = int(environ["LOCAL_RANK"])
	setups(device)
	ddp_setup()
 
	model, tokenizer, optimizer = load_model_and_tokenizer(device)
	model = DDP(model, device_ids=[device])

	# Loading datasets
	train_dataset = ShakespearDataset(tokenizer, TEXT_FILE)
	train_dataloader = DataLoader(
		train_dataset, batch_size=BATCH_SIZE, pin_memory=True, shuffle=False, sampler=DistributedSampler(train_dataset),
		collate_fn=partial(generate_ip_op_pairs, tokenizer=tokenizer, max_seq_len=MAX_SEQ_LEN)
	)

	# Training begins here.
	trainer = Trainer(model, tokenizer, optimizer, EPOCH, train_dataloader, MODEL_SAVE_DIR, CKPT_FILE, 
		{'DEVICE': device, 'DEVICE_TYPE': DEVICE_TYPE, 'USE_MIX_PRECISION': USE_MIX_PRECISION, 'GRADIENT_CLIP': 
		GRADIENT_CLIP, 'MAX_SEQ_LEN': MAX_SEQ_LEN}
	)
	trainer.train_ddp()

	destroy_process_group()

def train():
	device = "cuda:3"
	setups(device)
	model, tokenizer, optimizer = load_model_and_tokenizer(device)
	# Will compile the model once and hence all layers will run faster during forward calls.
	# model = torch.compile(model)
	
	# Loading datasets
	train_dataset = ShakespearDataset(tokenizer, TEXT_FILE) 
	train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
		collate_fn=partial(generate_ip_op_pairs, tokenizer=tokenizer, max_seq_len=MAX_SEQ_LEN))
	
	# Training begins here.
	trainer = Trainer(model, tokenizer, optimizer, EPOCH, train_dataloader, MODEL_SAVE_DIR, CKPT_FILE, 
		{'DEVICE': device, 'DEVICE_TYPE': DEVICE_TYPE, 'USE_MIX_PRECISION': USE_MIX_PRECISION, 
		'GRADIENT_CLIP': GRADIENT_CLIP}
	)
	trainer.train()


if __name__ == "__main__":
	cleanups()
	parser = ArgumentParser("Parser for training llms fundamentally.")
	parser.add_argument("--ddp", action="store_true", help="Run training with Distributed Data Parallel.")
	args = parser.parse_args()
	
	if args.ddp:
		print("Starting training with DDP")
		train_ddp()
	else:
		print("Starting training as single process")
		train()
	
	cleanups()
