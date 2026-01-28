# mini_llm

> A minimal, from‑scratch implementation of a tokenizer + transformer‑based language model, designed to learn *Cervantes‑style* text generation.

---

## ✨ Overview

**mini_llm** is a lightweight project that walks through the full pipeline of building a small language model:

1. **Text preprocessing and tokenization** (BPE)
2. **Training a transformer‑based LLM**
3. **Autoregressive text generation**

---

## 🧩 Tokenizer

The tokenizer is implemented from scratch and provides:

* **Byte Pair Encoding (BPE)** algorithm
* **Regex‑based preprocessing**, using the same pattern as GPT‑4
* Full workflow support:

  * Training
  * Encoding / decoding
  * Saving and loading tokenizer state

This allows the tokenizer to be trained independently and reused across experiments.

---

## 🤖 Language Model (LLM)

The LLM is a **transformer‑based autoregressive model** trained to generate text in the style of **Miguel de Cervantes**.

Features:

* Decoder‑only transformer architecture
* End‑to‑end training from raw text
* Characteristic literary text generation

> ⚠️ **Note**: Model checkpoint saving/loading is **not yet implemented**, but is planned as a future improvement.

---

## 🚀 Training the Model

To train the language model:

```bash
cd src
python train.py
```

Training configuration and hyperparameters can be modified directly in the training script.

---

## 🧪 Testing

Tokenizer functionality is covered by unit tests.

To run the tests:

```bash
pytest -v
```

---

## 🛠️ Roadmap

Planned improvements:

* [ ] Model checkpoint save/load support
* [ ] Inference‑only generation script
* [ ] Training metrics logging
* [ ] Improved documentation and examples

---