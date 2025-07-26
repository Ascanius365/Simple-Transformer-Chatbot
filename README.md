# Simple-Transformer-Chatbot

This project implements a decoder-only Transformer model in PyTorch for text generation.  
The model architecture is inspired by GPT-style transformers, but uses a pretrained **BERT tokenizer and embedding layer**.  
The 12 decoder layers are **randomly initialized** and trained from scratch on custom text data.
The model is trained on the **DailyDialog** dataset. 

# Features

- BERT tokenizer integration using Hugging Face's `transformers` library
- Positional embeddings
- Decoder Stack: 12 custom transformer decoder blocks, trained from scratch
- Causal self-attention
- Masked cross-entropy loss with ignored padding
- Learning rate scheduling (`ReduceLROnPlateau`)
- GPU support via `torch.device`
- Save/load model with `.pth` files
- Live plot generation with `matplotlib`
