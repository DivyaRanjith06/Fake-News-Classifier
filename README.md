# 📰 Fake News Classifier — NLP Project

An end-to-end NLP pipeline for detecting fake news using morphological analysis, POS-based linguistic features, TF-IDF, and BERT embeddings. Built on the **ISOT Fake News Dataset**, this project compares four progressively richer models to understand what linguistic signals separate real journalism from misinformation.

---

## 📌 Overview

Fake news detection is framed as a binary text classification task:
- **Label 0** → Unreliable / Fake
- **Label 1** → Reliable / Real

Rather than treating it as a pure bag-of-words problem, this project extracts **interpretable linguistic features** (superlative usage, pronoun ratio, dependency tree depth) and combines them with both classical TF-IDF and contextual BERT embeddings.

---

## 🧠 Models Compared

| Model | Features | Approach |
|---|---|---|
| **A** — TF-IDF Only | Bag-of-words n-grams | Baseline |
| **B** — TF-IDF + Linguistic | TF-IDF + 8 hand-crafted style features | Best interpretable model |
| **C** — BERT Only | DistilBERT [CLS] embeddings | Contextual semantics |
| **D** — BERT + Linguistic | BERT + 8 linguistic features | Best overall model |

---

## 📂 Repository Structure

```
fake-news-classifier/
│
├── FakeNewsClassifier__1_.ipynb    # Main notebook (full pipeline)
├── README.md                       # Project documentation
├── .gitignore                      # Files to exclude from version control

```

---

## 📊 Dataset

**ISOT Fake News Dataset** — Contains ~44,000 real-world news articles scraped from Reuters (real) and various unreliable sources flagged by Politifact (fake).

🔗 Download: [https://www.kaggle.com/datasets/csmalarkodi/isot-fake-news-dataset](https://www.kaggle.com/datasets/csmalarkodi/isot-fake-news-dataset)

Files needed:
- `Fake.csv`
- `True.csv`

> ⚠️ The notebook loads from Google Drive (`/content/drive/MyDrive/NLP/`). Update the paths in **Cell 5** to match your local setup or Drive folder.

---


## 🔍 Pipeline Summary

```
Raw CSVs (Fake + Real)
  → Sample 1000 articles → Label (0/1)

Phase 1 — Morphological Preprocessing
  → Sentence tokenization → Word tokenization
  → Stop-word removal (pronouns retained) → Lemmatization

Phase 2 — Feature Engineering via POS Tagging
  → Superlative ratio, Proper noun ratio, Pronoun ratio
  → Adjective ratio, Exclamation ratio, Question ratio

Phase 3 — Syntax Analysis
  → Avg sentence length → Dependency tree depth (spaCy)
  → Constituency tree height proxy → Mann-Whitney U test

Phase 4 — Classification
  → TF-IDF hyperparameter grid search
  → 4 models (A/B/C/D) trained & evaluated
  → Error analysis + Feature importance
  → t-SNE visualizations (TF-IDF space vs BERT space)
```

---

## 📈 Key Features

- **Linguistic feature engineering** — 8 hand-crafted POS-based features grounded in misinformation research
- **Hypothesis testing** — Mann-Whitney U test on dependency tree depth (real vs fake)
- **TF-IDF hyperparameter grid search** — 6 configurations compared across n-gram ranges and feature counts
- **DistilBERT embeddings** — Contextual [CLS] token representations via HuggingFace Transformers
- **Error analysis** — False positives and false negatives inspected with feature-level explanations
- **t-SNE visualization** — 2D projection of TF-IDF, BERT, and linguistic feature spaces

---

## 🔬 Linguistic Hypotheses Tested

| Feature | Hypothesis |
|---|---|
| Superlative ratio | Fake news uses more extreme language ("best", "worst") |
| Proper noun ratio | Real news names specific people and places |
| Pronoun ratio | Fake news uses more first-person language (bias signal) |
| Exclamation ratio | Fake news uses more emotional punctuation |
| Dependency tree depth | Real news has syntactically more complex sentences |

---
