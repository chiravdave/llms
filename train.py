from typing import Any, Tuple
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
from src.models.deepseek import DeepSeek
from src.dataloader import ShakespearDataset, generate_ip_op_pairs
from src.trainer import Trainer
from src.utils import (
    cleanups, configure_optimizer, load_train_config, ModelArgs, MistralModelArgs, DeepSeekModelArgs, setups
)


HF_MODEL_DIR = "/root/.cache/huggingface/hub"
MODEL_NAME = "deepseek" 							# Update model name here to run different model
TEXT_FILE = "data/input.txt"
TRAIN_CONFIG_FILE = "train_config.yaml"
DEVICE_TYPE = "cuda" if torch.cuda.is_available() else "cpu"


def ddp_setup():
	init_process_group(backend="nccl")
	torch.cuda.set_device(int(environ['LOCAL_RANK']))

def load_model_and_tokenizer(device: str) -> Tuple[nn.Module, AutoTokenizer, AdamW, ModelArgs]:
	train_config = load_train_config(TRAIN_CONFIG_FILE, MODEL_NAME)
	# Loading tokenizer.
	tokenizer = None
	if path.exists(train_config['save']['model_save_dir']):
		print("Tokenizer was loaded from the latest checkpoint.")
		tokenizer = AutoTokenizer.from_pretrained(train_config['save']['model_save_dir'], local_files_only=True)
	else:
		tokenizer = AutoTokenizer.from_pretrained(train_config['tokenizer'], cache_dir=HF_MODEL_DIR)
		tokenizer.add_special_tokens({'pad_token': '[PAD]'})
		makedirs(train_config['save']['model_save_dir'])
		tokenizer.save_pretrained(train_config['save']['model_save_dir'])

	# Loading model.
	model = None
	if MODEL_NAME == "llama2":
		print("Loading llama2 model")
		model_args = ModelArgs(vocab_size=len(tokenizer), device=device, **train_config['model_args'])
		model = Llama2(model_args).to(device)
	elif MODEL_NAME == "mistral":
		print("Loading mistral model")
		model_args = MistralModelArgs(vocab_size=len(tokenizer), device=device, **train_config['model_args'])
		model = Mistral(model_args).to(device)
	else:
		print("Loading deepseek model")
		model_args = DeepSeekModelArgs(vocab_size=len(tokenizer), device=device, **train_config['model_args'])
		model = DeepSeek(model_args).to(device)

	optimizer = configure_optimizer(model.named_parameters(), lr_rate=6e-4, weight_decay=0.1, device_type=DEVICE_TYPE)
	# Loading the model from previous checkpoint if present.
	if path.exists(path.join(train_config['save']['model_save_dir'], train_config['save']['ckpt'])):
		print("Model was loaded from the latest checkpoint.")
		model.load_ckpt(path.join(train_config['save']['model_save_dir'], train_config['save']['ckpt']), optimizer)

	return model, tokenizer, optimizer, train_config

def train_ddp():
	device = int(environ['LOCAL_RANK'])
	setups(device)
	ddp_setup()
 
	model, tokenizer, optimizer, train_config = load_model_and_tokenizer(device)
	model = DDP(model, device_ids=[device])

	# Loading datasets
	train_dataset = ShakespearDataset(tokenizer, TEXT_FILE)
	train_dataloader = DataLoader(
		train_dataset, 
		batch_size=train_config['hyper_params']['batch_size'], 
		pin_memory=True, 
		shuffle=False, 
		sampler=DistributedSampler(train_dataset),
		collate_fn=partial(
			generate_ip_op_pairs, tokenizer=tokenizer, max_seq_len=train_config['model_args']['max_seq_len']
	))

	# Training begins here.
	trainer = Trainer(
		model=model, 
		tokenizer=tokenizer, 
		optimizer=optimizer, 
		train_dataloader=train_dataloader, 
		model_save_dir=train_config['save']['model_save_dir'], 
		ckpt=train_config['save']['ckpt'],
		device=device, 
		device_type=DEVICE_TYPE,
		**train_config['hyper_params'],
	)
	trainer.train_ddp()

	destroy_process_group()

def train():
	device = "cuda:3"
	setups(device)
	model, tokenizer, optimizer, train_config = load_model_and_tokenizer(device)
	# Will compile the model once and hence all layers will run faster during forward calls.
	# model = torch.compile(model)
	
	# Loading datasets
	train_dataset = ShakespearDataset(tokenizer, TEXT_FILE) 
	train_dataloader = DataLoader(
		train_dataset, 
		batch_size=train_config['hyper_params']['batch_size'], 
		shuffle=True, 
		collate_fn=partial(
			generate_ip_op_pairs, tokenizer=tokenizer, max_seq_len=train_config['model_args']['max_seq_len']
	))
	
	# Training begins here.
	trainer = Trainer(
		model=model, 
		tokenizer=tokenizer, 
		optimizer=optimizer, 
		train_dataloader=train_dataloader,
		model_save_dir=train_config['save']['model_save_dir'], 
		ckpt=train_config['save']['ckpt'], 
		device=device, 
		device_type=DEVICE_TYPE, 
		**train_config['hyper_params'],
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
