from typing import List, Tuple

from torch import Tensor, roll
from torch.utils.data import Dataset

from transformers import AutoTokenizer


class ShakespearDataset(Dataset):
	def __init__(self, tokenizer: AutoTokenizer, max_seq_len: int, text_file: str):
		self.tokenizer = tokenizer
		self.max_seq_len = max_seq_len
		self.sentences = self._extract_training_sentences(text_file)

	def __len__(self) -> int:
		return len(self.sentences)

	def __getitems__(self, idxs: List[int]) -> Tuple[Tensor, Tensor]:
		batched_sentences = [self.sentences[idx] for idx in idxs]
		
		return self._generate_ip_op_pairs(batched_sentences)

	def _extract_training_sentences(self, text_file: str) -> List[str]:
		"""
		Generate sentences from the input file. Each sentence will have at max 100 words.
		"""
		sentences = list()
		with open(text_file, 'r') as f:
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

	def _generate_ip_op_pairs(self, sentences: List[str]) -> Tuple[Tensor, Tensor]:
		"""
		Creates input & output pair for text generation. Output tokens will be shifted by one position to the left.
		"""
		input_token_ids = None
		if len(sentences) == 1:
			input_token_ids = self.tokenizer(sentences, truncation=True, max_length=self.max_seq_len, 
				return_tensors="pt", return_attention_mask=False)['input_ids']
		else:
			input_token_ids = self.tokenizer(sentences, padding="longest", truncation=True, max_length=self.max_seq_len, 
				return_tensors="pt", return_attention_mask =False)['input_ids']

		# Generating target token ids having eos_token_id and shifted one position to the left.
		output_token_ids = roll(input_token_ids, -1, 1)
		output_token_ids[:, -1] = self.tokenizer.pad_token_id

		for ids in output_token_ids:
			first_pad_index = (ids == self.tokenizer.pad_token_id).nonzero(as_tuple=True)[0][0]
			ids[first_pad_index] = self.tokenizer.eos_token_id

		return input_token_ids, output_token_ids
