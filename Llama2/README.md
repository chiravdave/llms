# Llama-2
This repository includes a pytorch implementation of Llama-2 model. The dataset used in the training of the model is based
on dialogues from a play. The model is a miniature version of the actual Llama-2 due to infrastructure constraints.

* Epochs = 30
* Batch Size = 32
* Max Seq Len = 512
* Dim = 128
* Decoder Blocks = 2
* Total Heads = 8
* Total KV Heads = 2
* Vocab Size = 32001

# DDP Based Distributed Training
* Command to initiate training on single node with multiple GPUs: 
  **torchrun --standalone --nproc-per-node=<NUM_GPUS> ddp_train.py**

# Results

* Text Generation
```
Test Sample 1: 
<s> 
 CORIOLANUS:
 I am a subtleer, I amain to kiss my
 sonness's as good as good as is the worst
 Will piece of you, and that I should have heard,
 you are inclined.
 
 MENENIUS:
 I am bound to you.
 
 MENENIUS:
 I am bound:
 You are cause, sir, we are sure
 To prone.
 
 Both:
 I am aat of your own report.
 
 MENENIUS:
 I am a Roman,
 And you, I am not like your country,
 In the better now: you are no lesser than
 it seems to your country than you can,
 For the thing I have well as cruelty one
 As if you be remembered.
 
 MARCIUS:
</s>

Test Sample 2: 
<s> kind than
 she went, or borneed with-hors.
 
 QUEEN ELIZABETH:
 O, f lad, frown bravery.
 
 KING RICHARD III:
 Sweet father, love, love!
 
 QUEEN ELIZABETH:
 Shins, she says, and in the hateful duke.
 
 QUEEN ELIZABETH:
 You have spoke to-day early: take away.
 
 KING RICHARD III:
 So long hath been balm with your banishment,
 To make fellow of ripefellow
 And lay the white three-pable and open
</s>
```
