import torch

from transformers import AutoTokenizer

from train import LLAMA2_MODEL, CKPT_PATH
from model import ModelArgs, Transformer


BATCH_SIZE = 1
MAX_SEQ_LEN = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEST_SAMPLES = 1


def complete_prompt(prompt: str) -> None:
	"""
	This function will complete the input prompt.
	"""
	if DEVICE == "cuda":
		torch.set_default_dtype(torch.float16)

	tokenizer = AutoTokenizer.from_pretrained(LLAMA2_MODEL)
	extra_vocab = tokenizer.add_special_tokens({'pad_token': '[PAD]'})
	vocab_size = tokenizer.vocab_size + extra_vocab
	model_args = ModelArgs(vocab_size=vocab_size, device=DEVICE, max_batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN)
	model = Transformer(model_args).to(DEVICE)
	model.load_weights(CKPT_PATH)
	
	prompt_token_ids = tokenizer(prompt, truncation=True, max_length=MAX_SEQ_LEN, return_attention_mask =False)['input_ids']
	# seq_len will be 1 during testing.
	input_token_id = torch.full((BATCH_SIZE, 1), prompt_token_ids[0]).to(DEVICE)
	prompt_tokens_len, start_pos = len(prompt_token_ids), 0
	with torch.no_grad():
		# First need to populate the k & v vectors based on the provided prompt.
		while start_pos < prompt_tokens_len - 1:
			_ = model(input_token_id, start_pos)
			start_pos += 1
			input_token_id[0][0] = prompt_token_ids[start_pos]

		# Now will run the prompt completion logic.
		while True:
			logits = model(input_token_id, start_pos)
			next_token_id = torch.argmax(logits.reshape(-1))
			input_token_id[0][0] = next_token_id
			prompt_token_ids.append(next_token_id)
			start_pos += 1

			if next_token_id == tokenizer.eos_token_id or start_pos == MAX_SEQ_LEN:
				break

		print(f"Prompt Completion:\n{tokenizer.decode(prompt_token_ids)}")


def test() -> None:
	"""
	This function will randomly generate sentences.
	"""
	if DEVICE == "cuda":
		torch.set_default_dtype(torch.float16)

	tokenizer = AutoTokenizer.from_pretrained(LLAMA2_MODEL)
	extra_vocab = tokenizer.add_special_tokens({'pad_token': '[PAD]'})
	vocab_size = tokenizer.vocab_size + extra_vocab
	model_args = ModelArgs(vocab_size=vocab_size, device=DEVICE, max_batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN)
	model = Transformer(model_args).to(DEVICE)
	model.load_weights(CKPT_PATH)
	# seq_len will be 1 during testing.
	input_token_id = torch.full((BATCH_SIZE, 1), tokenizer.bos_token_id).to(DEVICE)

	# Testing starts here.
	for sample in range(1, TEST_SAMPLES + 1):
		output_token_ids, start_pos = [tokenizer.bos_token_id], 0
		with torch.no_grad():
			while True:
				logits = model(input_token_id, start_pos)
				next_token_id = torch.argmax(logits.reshape(-1))
				input_token_id[0][0] = next_token_id
				output_token_ids.append(next_token_id)
				start_pos += 1

				if next_token_id == tokenizer.eos_token_id or start_pos == MAX_SEQ_LEN:
					break

		print(f"Test Sample {sample}:\n{tokenizer.decode(output_token_ids)}\n")


if __name__ == "__main__":
	test()
	complete_prompt("First Citizen:We are accounted poor citizens, the patricians good.\nWhat authority surfeits on would relieve us: if they")