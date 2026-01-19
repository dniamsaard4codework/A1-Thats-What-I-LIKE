# A1: That's What I LIKE

**NLU Assignment 1** — Word Embedding Models  
**Student**: Dechathon Niamsa-ard [st126235]

---

## Overview

This project implements Word2Vec and GloVe word embedding models from scratch, following the original papers. Models are trained on the NLTK Brown Corpus (news category) and evaluated using standard benchmarks for semantic/syntactic accuracy and human similarity judgments.

---

## Project Structure

```
├── app/                              # Flask web application
│   ├── app.py                       # Main Flask server
│   └── templates/index.html         # Search interface
├── dataset/                          # Evaluation datasets
│   ├── word-test.v1.txt             # Word analogy test
│   └── wordsim353crowd/             # Human similarity judgments
├── lab_01/                           # Lab reference notebooks
├── model/                            # Saved model weights (.pt)
├── st126235_assignment_1.ipynb       # Main notebook (training + evaluation)
├── pyproject.toml                    # Project config
└── requirements.txt                  # Dependencies
```

---

## Quick Start

```bash
# Setup environment
uv venv && uv sync
# Or: pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('brown'); nltk.download('punkt')"

# Train models (run notebook cells)
jupyter notebook st126235_assignment_1.ipynb

# Run web app
cd app && python app.py
```

Open http://localhost:5000 in your browser.

---

## Task 1: Preparation and Training (3 points)

**Objective**: Implement Word2Vec and GloVe models based on the original papers.

### What I Did:
- Read and studied the Word2Vec and GloVe papers to understand the math
- Built 3 models from scratch using PyTorch:
  - **Skipgram** — Standard Word2Vec with full softmax
  - **Skipgram NEG** — Word2Vec with negative sampling (faster training)
  - **GloVe** — Global Vectors using co-occurrence matrix
- Used **Brown Corpus (news category)** from NLTK as training data
- Created a **dynamic window size function** — can adjust context window during training (default = 2)
- All models use 50-dimensional embeddings and train for 1000 epochs

### Training Details:
- Corpus: ~4,600 sentences from Brown news articles
- Vocabulary: ~10,000 unique words
- Batch size: 128
- Negative samples (for NEG model): 5

---

## Task 2: Model Comparison and Analysis (3 points)

**Objective**: Compare models on training metrics and standard NLP benchmarks.

### Comparison Results:

| Model | Window | Loss | Time | Syntactic Acc | Semantic Acc |
|-------|--------|------|------|---------------|--------------|
| Skipgram | 2 | 19.80 | 375s | 0.00% | 0.00% |
| Skipgram NEG | 2 | 6.44 | 344s | 0.00% | 0.00% |
| GloVe | 2 | 1038.03 | 10s | 0.00% | 0.00% |
| GloVe (Gensim) | - | - | - | 55.45% | 93.87% |

### Similarity Correlation (WordSim353):

| Model | Spearman Corr | MSE |
|-------|---------------|-----|
| Skipgram | ~0.15 | 0.1543 |
| Skipgram NEG | ~0.15 | 0.1537 |
| GloVe | ~0.16 | 0.1589 |
| GloVe (Gensim) | ~0.60 | 0.0441 |

### Key Findings:
- **0% accuracy is expected** — our small corpus (~4.6k sentences) doesn't contain most analogy words like country capitals
- **Gensim pre-trained model** shows much better results because it's trained on billions of words (Wikipedia + Gigaword)
- **GloVe trains fastest** due to the matrix factorization approach
- **Negative sampling** achieves lower loss than full softmax while being faster
- The notebook includes **PCA visualizations** showing how words cluster in embedding space

---

## Task 3: Web Application (2 points)

**Objective**: Build a search interface to find similar contexts using word embeddings.

### Features:
- **Flask web app** with clean, minimal UI
- **Search box** — enter any query to find similar sentences
- **Model switcher** — choose between Skipgram, Skipgram NEG, GloVe, or Gensim GloVe
- **Top 10 results** displayed with similarity scores
- Uses **dot product** between query embedding and corpus sentence embeddings

### How It Works:
1. User enters a search query (e.g., "government policy")
2. Query is converted to embedding (average of word vectors)
3. Each corpus sentence is also converted to an embedding
4. Dot product computes similarity
5. Top 10 most similar sentences returned

### Running the App:
```bash
cd app
python app.py
# Open http://localhost:5000
```

---

## Visualizations

The notebook includes:
- **Training loss plots** for all 3 models
- **PCA 2D projections** showing word clustering
- **Detailed similarity tables** comparing model predictions to human judgments

---

## Dataset Sources

| Dataset | Description | Source |
|---------|-------------|--------|
| Brown Corpus | News articles (NLTK) | [NLTK Data](https://www.nltk.org/nltk_data/) |
| Word Analogies | word-test.v1.txt | [GitHub](https://raw.githubusercontent.com/nicholas-leonard/word2vec/master/questions-words.txt) |
| WordSim353 | Human similarity ratings | [Kaggle](https://www.kaggle.com/datasets/julianschelb/wordsim353-crowd) |

**Citation**: Brown Corpus — Francis, W., & Kucera, H. (1979). Brown Corpus Manual.

**Information**: The link of WordSim353 in the pdf do not work: http://alfonseca.org/eng/research/wordsim353.html
---

## Notes

- The low accuracy on analogy tasks is **expected behavior** for a small corpus
- Compare with Gensim pre-trained to see how larger training data helps
- Models are saved in `model/` folder and loaded by the Flask app

---

## Usage

### Web Interface

1. Enter a search query (e.g., "government policy")
2. Select a model from the buttons
3. View top-10 similar contexts with similarity scores

### API Endpoints

- `GET /` — Main search interface
- `POST /search` — Search for similar contexts
- `POST /switch_model` — Switch between models

---

## Requirements

- Python 3.11+
- PyTorch 2.0+
- Flask 3.0+
- Gensim 4.3+
- NLTK 3.8+

See `requirements.txt` for full list.

---