from typing import Tuple
from os import path

import torch
from transformers import AutoTokenizer

from src.models.llama2 import Llama2
from src.models.mistral import Mistral
from src.utils import ModelArgs
from src.utils import (
	cleanups, load_train_config, ModelArgs, MistralModelArgs, DeepSeekModelArgs, setups
)
from src.models.llama2 import Llama2
from src.models.mistral import Mistral
from src.models.deepseek import DeepSeek


TRAIN_CONFIG_FILE = "train_config.yaml"
MODEL_NAME = "deepseek" 							# Update model name here to run different model
DEVICE_TYPE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "cuda:3"
DTYPE = torch.bfloat16


def load_model_and_tokenizer() -> Tuple[Llama2, AutoTokenizer, ModelArgs]:
	test_config = load_train_config(TRAIN_CONFIG_FILE, MODEL_NAME)
	tokenizer = AutoTokenizer.from_pretrained(test_config['save']['model_save_dir'], local_files_only=True)
	# Loading model.
	model = None
	if MODEL_NAME == "llama2":
		print("Loading llama2 model")
		model_args = ModelArgs(vocab_size=len(tokenizer), device=DEVICE, **test_config['model_args'])
		model = Llama2(model_args).to(DEVICE)
	elif MODEL_NAME == "mistral":
		print("Loading mistral model")
		model_args = MistralModelArgs(vocab_size=len(tokenizer), device=DEVICE, **test_config['model_args'])
		model = Mistral(model_args).to(DEVICE)
	else:
		print("Loading deepseek model")
		model_args = DeepSeekModelArgs(vocab_size=len(tokenizer), device=DEVICE, **test_config['model_args'])
		model = DeepSeek(model_args).to(DEVICE)

	model.load_ckpt(path.join(test_config['save']['model_save_dir'], test_config['save']['ckpt']))

	return model, tokenizer, test_config

def completions() -> None:
	"""
	This function will complete the input prompt.
	"""
	model, tokenizer, test_config = load_model_and_tokenizer()
	generation_config = {
	'TEMPERATURE': 1.0,
	'USE_CACHE': True,
	'EOS_TOKEN_ID': tokenizer.eos_token_id,
	'MAX_BATCH_SIZE': 1,
	'MAX_SEQ_LEN': 512,
	}
	model.eval(generation_config['MAX_BATCH_SIZE'], generation_config['MAX_SEQ_LEN'])
	while True:
		prompt = input("Enter your prompt: ")
		token_ids = tokenizer(
			prompt, truncation=True, max_length=test_config['model_args']['max_seq_len'], return_tensors="pt", 
			return_attention_mask=False
		)['input_ids'].to(DEVICE)
		with torch.no_grad():
			with torch.autocast(
				device_type=DEVICE_TYPE, dtype=DTYPE, enabled=test_config['hyper_params']['use_mix_precision']
			):
				# Now will run the prompt completion logic.
				output_tokens = model.generate(token_ids, **generation_config)[0]
				print(f"Input prompt:\n{prompt}\n")
				print(f"Completion:\n{tokenizer.decode(output_tokens, skip_special_tokens=True)}")

		selection = input("Enter 'c' to continue or 'q' to quit")
		if selection == "q":
			break


if __name__ == "__main__":
	cleanups()
	setups(DEVICE_TYPE)
	completions()
	cleanups()