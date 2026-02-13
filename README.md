# LLMs From Scratch [![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/chiravdave/llms)

This repository provides a from-scratch implementation, training, and testing setup for state-of-the-art Large Language 
Models (LLMs), including LLaMA-2 and Mistral (Base + Mixture of Experts). It also explores advanced optimization and distributed training techniques used in modern LLM pipelines.


## Implemented Models

* Llama-2
* Mistral
* DeepSeek-V2

## Optimization Techniques Used

* Model Compilation: For improved runtime performance.
* Mixed Precision Training: Reduces memory usage and increases speed using `torch.float16/bfloat16`.
* Flash Attention: High-performance attention mechanism for transformer models.

## Distributed Training
* DDP: Leveraged for efficient multi-GPU training with PyTorch

## Project Structure

```
llms/
│
├── data/
│   └── input.txt      		# Training Dataset
│
├── results/
│   ├── DeepSeek.md         # DeepSeek results
│   ├── Llama2.md           # Llama-2 results
│   └── Mistral.md          # Mistral results
│
├── src/
│	├── layers/
│	│   ├── attention.py    # Different Attention Layers Implementation
│   │   └── norm.py         # Different Normalization Layers Implementation
│	├── models/
│   │	├── deepseek.py   	# DeepSeek model implementation
│   │   ├── llama2.py 		# Llama-2 model implementation
│   │   └── mistral.py      # Mistral base and MoE models implementation
│   │
│   ├── dataloader.py       # Custom Dataloader   
│   ├── trainer.py        	# Generic training loop
│   └── utils.py       		# Utility functions
│
├── train.py   		   		# Training Script
├── test.py            		# Testing Script
├── train_config.yaml       # Training Config 
│
└── README.md          		# Project Documentation
```

## Running & Testing Instructions

### Running As A Single Process

```bash
python3 train.py
```

**NOTE** Edit training parameters inside `train.py` & `train_config.yaml` as per the need before going for a run.

### Testing

```bash
python3 test.py
```

**NOTE**: Edit other parameters inside `test.py` & `train_config.yaml` as per the need before going for the run.

### DDP Based Distributed Training

* Single node multi-gpu setup
```bash
torchrun --standalone --nproc-per-node=<NUM_GPUS> train.py --ddp
```

**NOTE**: Edit other parameters inside `train.py` & `train_config.yaml` as per the need before going for a run.
