
# WordWave – Next Word & Sequence Predictor

[![Made with Python](https://img.shields.io/badge/Made%20with-Python-3670A0?logo=python&logoColor=white)](https://www.python.org/)
[![Deep Learning](https://img.shields.io/badge/Model-PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Streamlit App](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

WordWave is an intelligent next-word and short-sequence predictor built on a **PyTorch Bidirectional LSTM** with an **attention mechanism**. The app provides real-time word generation and metric-based evaluation through a user-friendly **Streamlit dashboard**.

---

- Built using a deep **Embedding → BiLSTM → Attention → Dense** pipeline for next-word prediction
- Supports **beam search decoding** to improve generation quality over greedy search
- Evaluates with key metrics:
  - **Top-5 Accuracy**
  - **Perplexity**
  - **BLEU Score**
- User can input a **seed sentence and target length**, and the app generates fluent text
- Designed as a **Streamlit web app** for easy interaction and visualization
- Trains and stores artifacts as **.pt** and **.pth** files

---

## Project Structure

```sh
word-wave/
├── app.py                 # Thin Streamlit entrypoint
├── config.yaml            # Static runtime and training settings
├── pyproject.toml         # Modern package metadata and dependency ranges
├── src/                   # Application package
│   ├── artifacts.py       # Model/tokenizer loading
│   ├── corpus.py          # Corpus discovery and streaming file iteration
│   ├── data.py            # Dataset preparation helpers
│   ├── generation.py      # Beam search and BLEU helpers
│   ├── metrics.py         # Model evaluation helpers
│   ├── model.py           # PyTorch BiLSTM attention model
│   ├── tokenizer.py       # Vocabulary utilities
│   ├── settings.py        # YAML settings loader
│   ├── train.py           # Training entrypoint
│   └── ui.py              # Streamlit rendering
├── README.md              # This file
└── LICENSE
```

## Configuration

The static application settings live in `config.yaml`. It defines the model and tokenizer paths, default UI values, supported plaintext file extensions, and the training split fractions used by the streaming corpus loader.

---

## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/ang-1107/word-wave.git
cd word-wave
```

### 2. Create a Virtual Environment and Install dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

All static paths, defaults, and training parameters live in `config.yaml`.

### 3. Train the model
Provide a plain-text file or a directory of plaintext files and train the model to create the model and tokenizer artifacts.

```bash
python -m src.train --data-path path/to/corpus_or_dir --epochs 5 --max-len 20
```

If you pass a directory, the trainer walks it recursively and includes all supported plaintext files. Training uses streamed batches and keeps train, validation, and test splits separate during epochs.

You can adjust `--epochs`, `--max-len`, `--max-vocab-size`, and the model size flags to fit your corpus.

### 4. Run the Streamlit app
```bash
streamlit run app.py
```

You’ll be able to enter text, pick how many words to generate, and see live predictions along with evaluation metrics.

---

## Model Overview

- **Framework**: PyTorch
- **Architecture**:
  `Embedding → Bidirectional LSTM → Attention → Dense`
- **Loss**: CrossEntropyLoss
- **Optimizer**: Adam
- **Evaluation Metrics**:
  - Top-5 Accuracy on the saved validation split
  - BLEU Score
  - Perplexity

- **Decoding**: Supports both **greedy** and **beam search** decoding
- **Training Data**: Any supported plaintext file or directory of plaintext files

---

## Sample Generation

```text
Seed: "deep learning models are"
Generated: "deep learning models are used to perform various tasks including natural language processing"
```

- BLEU Score: 0.38
- Perplexity: 215.4

---

## Evaluation

The Streamlit app shows metrics computed from the validation split saved alongside the trained PyTorch model.

### Top-5 Accuracy

Measures how often the correct next token appears in the model’s top 5 predictions.

### BLEU Score

Compares generated text to a reference sentence using the `nltk` BLEU metric.

### Perplexity

Calculated from the average cross-entropy over the validation split; lower is better.

---

## Future Improvements

- Add character-level prediction
- Fine-tune with larger dataset portions
- Integrate GPT-style transformer decoder for comparison
- Export as REST API for backend integration
