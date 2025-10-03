# LLMs From Scratch

This repository provides a from-scratch implementation, training, and testing setup for state-of-the-art Large Language 
Models (LLMs), including LLaMA-2 and Mistral (Base + Mixture of Experts). It also explores advanced optimization and distributed training techniques used in modern LLM pipelines.


## Implemented Models

* Llama-2
* Mistral

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
│   ├── Llama2.md           # LLaMA-2 results
│   └── Mistral.md          # Mistral results
│
├── src/
│	├── models/
│   │	├── llama2.py   	# LLaMA-2 model implementation
│   │   └── mistral.py 		# Mistral base and MoE models implementation
│   │
│   ├── dataloader.py       # Custom Dataloader   
│   ├── trainer.py        	# Generic training loop
│   └── utils.py       		# Utility functions
│
├── train.py   		   		# Training Script
├── test.py            		# Testing Script
│
└── README.md          		# Project Documentation
```

## Running & Testing Instructions

### Running As A Single Process

```bash
python3 train.py
```

**NOTE** Edit other parameters inside `train.py` as per the need before going for a run.

### Testing

```bash
python3 test.py
```

**NOTE** Edit other parameters inside `test.py` as per the need before going for the run.

### DDP Based Distributed Training

* Single node multi-gpu setup
```bash
torchrun --standalone --nproc-per-node=<NUM_GPUS> train.py --ddp
```

**NOTE** Edit other parameters inside `train.py` as per the need before going for a run.