"""
Flask Web Application for Word Embedding Search
A1: That's What I LIKE - Task 3

This application allows users to search for similar contexts using trained word embeddings.
Supports 4 models: Skipgram, Skipgram NEG, GloVe, and GloVe (Gensim)
"""

from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
import numpy as np
import os
import sys
import threading
import gensim.downloader as api

app = Flask(__name__)

# Model loading status
model_loading = False
model_load_error = None

# Model path
MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model"
)


class Skipgram(nn.Module):
    """Standard Skipgram Word2Vec model with softmax."""

    def __init__(self, voc_size, emb_size):
        super(Skipgram, self).__init__()
        self.embedding_center = nn.Embedding(voc_size, emb_size)
        self.embedding_outside = nn.Embedding(voc_size, emb_size)

    def forward(self, center, outside, all_vocabs):
        pass  # Not needed for inference

    def get_embedding(self, word_idx):
        """Get word embedding by averaging center and outside embeddings."""
        word_tensor = torch.LongTensor([word_idx])
        embed_c = self.embedding_center(word_tensor)
        embed_o = self.embedding_outside(word_tensor)
        return ((embed_c + embed_o) / 2).detach().cpu().numpy().flatten()


class SkipgramNeg(nn.Module):
    """Skipgram Word2Vec with Negative Sampling."""

    def __init__(self, voc_size, emb_size):
        super(SkipgramNeg, self).__init__()
        self.embedding_center = nn.Embedding(voc_size, emb_size)
        self.embedding_outside = nn.Embedding(voc_size, emb_size)
        self.logsigmoid = nn.LogSigmoid()

    def forward(self, center, outside, negative):
        pass  # Not needed for inference

    def get_embedding(self, word_idx):
        """Get word embedding by averaging center and outside embeddings."""
        word_tensor = torch.LongTensor([word_idx])
        embed_c = self.embedding_center(word_tensor)
        embed_o = self.embedding_outside(word_tensor)
        return ((embed_c + embed_o) / 2).detach().cpu().numpy().flatten()


class Glove(nn.Module):
    """GloVe model implementation."""

    def __init__(self, voc_size, emb_size):
        super(Glove, self).__init__()
        self.center_embedding = nn.Embedding(voc_size, emb_size)
        self.outside_embedding = nn.Embedding(voc_size, emb_size)
        self.center_bias = nn.Embedding(voc_size, 1)
        self.outside_bias = nn.Embedding(voc_size, 1)

    def forward(self, center, outside, coocs, weighting):
        pass  # Not needed for inference

    def get_embedding(self, word_idx):
        """Get word embedding by averaging center and outside embeddings."""
        word_tensor = torch.LongTensor([word_idx])
        embed_c = self.center_embedding(word_tensor)
        embed_o = self.outside_embedding(word_tensor)
        return ((embed_c + embed_o) / 2).detach().cpu().numpy().flatten()


class GensimGloveWrapper:
    """Wrapper for Gensim GloVe model to provide same interface as PyTorch models."""

    def __init__(self, gensim_model):
        self.gensim_model = gensim_model

    def get_embedding(self, word):
        """Get word embedding from gensim model."""
        try:
            return self.gensim_model[word]
        except KeyError:
            return None


# Global variables for loaded model
model = None
word2index = None
index2word = None
corpus = None
model_name = None
is_gensim = False

# Pre-cached Gensim model to avoid reload delay
_gensim_glove_cache = None


def preload_gensim_model():
    """Pre-load Gensim GloVe model in background to avoid first-search delay."""
    global _gensim_glove_cache
    try:
        print("Pre-loading Gensim GloVe model in background...")
        _gensim_glove_cache = api.load("glove-wiki-gigaword-100")
        print("Gensim GloVe model pre-loaded successfully!")
    except Exception as e:
        print(f"Failed to pre-load Gensim model: {e}")


def load_model(model_type="skipgram"):
    """Load a trained model from the model directory."""
    global model, word2index, index2word, corpus, model_name, is_gensim
    global model_loading, model_load_error, _gensim_glove_cache

    model_loading = True
    model_load_error = None

    # Handle gensim model separately
    if model_type == "gensim_glove":
        try:
            print("Loading Gensim GloVe model (glove-wiki-gigaword-100)...")
            # Use cached model if available, otherwise load fresh
            if _gensim_glove_cache is not None:
                gensim_model = _gensim_glove_cache
                print("Using pre-cached Gensim model.")
            else:
                gensim_model = api.load("glove-wiki-gigaword-100")
                _gensim_glove_cache = gensim_model  # Cache for next time
            
            model = GensimGloveWrapper(gensim_model)
            is_gensim = True
            model_name = "gensim_glove"

            # Load corpus from skipgram model for searching
            skipgram_file = os.path.join(MODEL_DIR, "skipgram.pt")
            if os.path.exists(skipgram_file):
                checkpoint = torch.load(skipgram_file, map_location="cpu")
                word2index = None  # Not used for gensim
                index2word = None
                corpus = checkpoint["corpus"]

            print("Gensim GloVe model loaded successfully!")
            model_loading = False
            return True
        except Exception as e:
            print(f"Error loading Gensim model: {e}")
            model_load_error = str(e)
            model_loading = False
            return False

    model_files = {
        "skipgram": "skipgram.pt",
        "skipgram_neg": "skipgram_neg.pt",
        "glove": "glove.pt",
    }

    model_file = os.path.join(MODEL_DIR, model_files.get(model_type, "skipgram.pt"))

    if not os.path.exists(model_file):
        print(f"Model file not found: {model_file}")
        print("Please run tester.ipynb first to train and save the models.")
        return False

    try:
        checkpoint = torch.load(model_file, map_location="cpu")
        word2index = checkpoint["word2index"]
        index2word = checkpoint["index2word"]
        corpus = checkpoint["corpus"]
        embedding_size = checkpoint["embedding_size"]
        voc_size = len(checkpoint["vocabs"])

        if model_type == "skipgram":
            model = Skipgram(voc_size, embedding_size)
        elif model_type == "skipgram_neg":
            model = SkipgramNeg(voc_size, embedding_size)
        else:
            model = Glove(voc_size, embedding_size)

        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        model_name = model_type
        is_gensim = False

        print(f"Loaded {model_type} model successfully!")
        print(f"Vocabulary size: {voc_size}")
        print(f"Corpus sentences: {len(corpus)}")
        model_loading = False
        return True

    except Exception as e:
        print(f"Error loading model: {e}")
        model_load_error = str(e)
        model_loading = False
        return False


def compute_sentence_embedding(sentence):
    """Compute sentence embedding as average of word embeddings."""
    if model is None:
        return None

    if is_gensim:
        # For gensim model, use word strings directly
        words = [w.lower() for w in sentence]
        embeddings = []
        for w in words:
            emb = model.get_embedding(w)
            if emb is not None:
                embeddings.append(emb)
        if not embeddings:
            return None
        return np.mean(embeddings, axis=0)
    else:
        # For PyTorch models, use word2index
        words = [w.lower() for w in sentence if w.lower() in word2index]
        if not words:
            return None
        embeddings = [model.get_embedding(word2index[w]) for w in words]
        return np.mean(embeddings, axis=0)


def search_similar_contexts(query, top_k=10):
    """Search for top-k most similar contexts using dot product."""
    if model is None or corpus is None:
        return []

    # Tokenize query
    query_words = query.lower().split()
    query_embed = compute_sentence_embedding(query_words)

    if query_embed is None:
        return []

    # Compute similarity with all sentences
    similarities = []
    for i, sent in enumerate(corpus):
        sent_embed = compute_sentence_embedding(sent)
        if sent_embed is not None:
            # Use dot product for similarity
            similarity = float(np.dot(query_embed, sent_embed))
            similarities.append(
                {"rank": 0, "score": similarity, "text": " ".join(sent)}
            )

    # Sort by similarity and get top-k
    similarities.sort(key=lambda x: x["score"], reverse=True)

    # Add ranks
    results = []
    for i, item in enumerate(similarities[:top_k], 1):
        item["rank"] = i
        results.append(item)

    return results


@app.route("/")
def index():
    """Main page with search interface."""
    return render_template("index.html", model_name=model_name)


@app.route("/model_status", methods=["GET"])
def model_status():
    """Check if model is ready."""
    return jsonify({
        "ready": model is not None and not model_loading,
        "loading": model_loading,
        "model": model_name,
        "error": model_load_error
    })


@app.route("/search", methods=["POST"])
def search():
    """Handle search queries."""
    data = request.get_json()
    query = data.get("query", "")

    if not query.strip():
        return jsonify({"error": "Please enter a search query", "results": []})

    if model_loading:
        return jsonify(
            {"error": "Model is still loading. Please wait a moment.", "results": []}
        )

    if model is None:
        return jsonify(
            {"error": "Model not loaded. Please train the model first.", "results": []}
        )

    results = search_similar_contexts(query)

    if not results:
        return jsonify(
            {
                "error": "No matching contexts found. Try different keywords.",
                "results": [],
            }
        )

    return jsonify({"results": results, "query": query})


@app.route("/switch_model", methods=["POST"])
def switch_model():
    """Switch to a different model."""
    data = request.get_json()
    model_type = data.get("model", "skipgram")

    if load_model(model_type):
        return jsonify({"success": True, "model": model_type})
    else:
        return jsonify({"success": False, "error": "Failed to load model"})


if __name__ == "__main__":
    print("=" * 60)
    print("Word Embedding Search - Flask Application")
    print("=" * 60)

    # Try to load the default model
    if load_model("skipgram"):
        # Pre-load Gensim model in background thread to avoid first-switch delay
        preload_thread = threading.Thread(target=preload_gensim_model, daemon=True)
        preload_thread.start()
        
        print("\nStarting web server...")
        print("Open http://localhost:5000 in your browser")
        print("=" * 60)
        app.run(debug=True, host="0.0.0.0", port=5000)
    else:
        print("\nNo trained model found!")
        print("Please run tester.ipynb first to train and save the models.")
        print("After training, run this script again.")
