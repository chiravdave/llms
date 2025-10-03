from typing import Tuple

import torch
from transformers import AutoTokenizer

from src.models.llama2 import Llama2
from src.models.mistral import Mistral
from src.utils import ModelArgs
from src.utils import cleanups, setups


MAX_SEQ_LEN = 512
MODEL_SAVE_DIR = "trained-models/small-mistral"
DEVICE_TYPE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "cuda:3"
MIX_PRECISION = True
DTYPE = torch.bfloat16
TEMPERATURE = 1.0
MODEL_NAME = "mistral"
MODEL_CKPT = "trained-models/small-mistral/model.pt"


def load_model_and_tokenizer() -> Tuple[Llama2, AutoTokenizer]:
	tokenizer = AutoTokenizer.from_pretrained(MODEL_SAVE_DIR, local_files_only=True)
	model_args = ModelArgs(vocab_size=len(tokenizer), device=DEVICE, max_batch_size=1, max_seq_len=MAX_SEQ_LEN)
	model = None
	if MODEL_NAME == "llama2":
			model = Llama2(model_args).to(DEVICE)
	elif MODEL_NAME == "mistral":
		model = Mistral(model_args).to(DEVICE)
	else:
		print(f"Invalid model name provided: {MODEL_NAME}")
		exit()

	model.load_ckpt(MODEL_CKPT)
	model.eval()

	return model, tokenizer

def completions() -> None:
	"""
	This function will complete the input prompt.
	"""
	model, tokenizer = load_model_and_tokenizer()
	while True:
		prompt = input("Enter your prompt: ")
		token_ids = tokenizer(
			prompt, truncation=True, max_length=MAX_SEQ_LEN, return_tensors="pt", return_attention_mask=False
		)['input_ids'].to(DEVICE)
		with torch.no_grad():
			with torch.autocast(device_type=DEVICE_TYPE, dtype=DTYPE, enabled=MIX_PRECISION):
				# Now will run the prompt completion logic.
				output_tokens = model.generate(token_ids, tokenizer.eos_token_id, TEMPERATURE)[0]
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