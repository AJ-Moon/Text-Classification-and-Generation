# Text Classification x Text Generation


## 📘 Part 1: CLIP — Contrastive Language–Image Pretraining

### 🧠 Concept

Implements a simplified version of CLIP (by OpenAI) to connect images and natural language through a shared embedding space. It uses the Flickr8k dataset for training.

### 🔧 Key Components

* **Config Class:** Hyperparameters and model settings
* **CLIPDataset:** Loads images and captions; tokenizes text using DistilBERT
* **Encoders:**

  * Image encoder: ResNet50 (via `timm`)
  * Text encoder: DistilBERT (via Huggingface)
* **Training Objective:** Align image and text embeddings

### 📦 Dataset

* **Flickr8k Dataset** from Kaggle: [Link](https://www.kaggle.com/datasets/adityajn105/flickr8k)

### 📚 Libraries

* PyTorch, Transformers, Torchvision, Timm, PIL, Pandas, Matplotlib, tqdm

---

## 🤖 Part 2: Transformers — Movie Script Language Model

### 🎬 Theme

A storytelling-themed implementation based on the Transformers franchise. Focuses on building a Transformer-based language model from scratch.

### 🧠 Architecture

* Encoder-Decoder (Vaswani et al. — "Attention Is All You Need")
* No recurrence; uses self-attention + positional encoding

### 🔧 Key Modules

* **TokenizedDataset:** Tokenizes and batches the movie script
* **Configuration:** Defines embedding dimensions, heads, blocks, etc.
* **Multi-Head Attention:** Causal self-attention for text generation
* **Feedforward Network:** GELU-activated MLP
* **Block:** Attention + FFN + LayerNorm
* **TransformerLM:** Complete stack for training/generation

### 🧪 Task

* Train on a *Transformers* movie script to generate character-style text

### 📚 Libraries

* PyTorch, `datasets`, `tiktoken`, `nltk`, numpy, matplotlib

---

## 📦 How to Run

1. Download the datasets (Flickr8k and movie script)
2. Open each notebook
3. Follow the cells from top to bottom — all key functions and classes are defined within

---


> "Transformers don’t just change language—they change the world of AI."
