from os import environ, makedirs, path

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from transformers import AutoTokenizer

from model import ModelArgs, Llama2
from dataloader import ShakespearDataset
from trainer import Trainer
from utils import cleanups, setups, configure_optimizer


LLAMA2_TOKENIZER = "meta-llama/Llama-2-7b-hf"
MODEL_SAVE_DIR = "models/llama2-7b-hf"
CKPT_FILE = "model.pt"
TEXT_FILE = "input.txt"
EPOCH = 50
BATCH_SIZE = 256
GRADIENT_CLIP = 1.0
MAX_SEQ_LEN = 512
USE_MIX_PRECISION = True
DEVICE_TYPE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = int(environ["LOCAL_RANK"])


def ddp_setup():
	init_process_group(backend="nccl")
	torch.cuda.set_device(int(environ["LOCAL_RANK"]))


def train():
	ddp_setup()
	setups(DEVICE_TYPE)

	# Loading tokenizer.
	tokenizer = None
	if path.exists(MODEL_SAVE_DIR):
		print("Tokenizer was loaded from the latest checkpoint.")
		tokenizer = AutoTokenizer.from_pretrained(MODEL_SAVE_DIR, local_files_only=True)
	else:
		tokenizer = AutoTokenizer.from_pretrained(LLAMA2_TOKENIZER, cache_dir="models")
		tokenizer.add_special_tokens({'pad_token': '[PAD]'})
		makedirs(MODEL_SAVE_DIR)
		tokenizer.save_pretrained(MODEL_SAVE_DIR)

	# Loading model & optimizer.
	model_args = ModelArgs(vocab_size=len(tokenizer), device=DEVICE, max_batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN)
	model = Llama2(model_args).to(DEVICE)
	optimizer = configure_optimizer(model.named_parameters(), lr_rate=6e-4, weight_decay=0.1, device_type=DEVICE_TYPE)
	# Loading the model from previous checkpoint if present.
	if path.exists(path.join(MODEL_SAVE_DIR, CKPT_FILE)):
		print("Model was loaded from the latest checkpoint.")
		model.load_ckpt(optimizer, path.join(MODEL_SAVE_DIR, CKPT_FILE))
	
	model = DDP(model, device_ids=[DEVICE])
	# Loading datasets
	train_dataset = ShakespearDataset(tokenizer, MAX_SEQ_LEN, TEXT_FILE)
	train_dataloader = DataLoader(
		train_dataset, batch_size=BATCH_SIZE, pin_memory=True, shuffle=False, sampler=DistributedSampler(train_dataset)
	)

	# Training begins here.
	trainer = Trainer(model, tokenizer, optimizer, EPOCH, train_dataloader, MODEL_SAVE_DIR, CKPT_FILE, 
		{'DEVICE': DEVICE, 'DEVICE_TYPE': DEVICE_TYPE, 'USE_MIX_PRECISION': USE_MIX_PRECISION, 'GRADIENT_CLIP': GRADIENT_CLIP,
		'MAX_SEQ_LEN': MAX_SEQ_LEN}
	)
	trainer.train_ddp()

	# model.save_ckpt(epoch, optimizer, path.join(MODEL_SAVE_DIR, "final_model.pt"))
	destroy_process_group()


if __name__ == "__main__":
	train()
