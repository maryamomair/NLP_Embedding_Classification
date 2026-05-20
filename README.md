# NLP Semantic Embedding & Feature Classification Pipeline

## Overview
This repository contains an end-to-end Natural Language Processing (NLP) pipeline designed to ingest unstructured text, engineer advanced semantic word embeddings, and evaluate supervised machine learning classifiers. 

Developed as part of advanced doctoral research, this codebase bridges the gap between raw textual logs and rigorous statistical inference. The workflows demonstrated here—specifically around information extraction, text representation, and high-dimensional classification—are directly applicable to data teams tracking technology domains, processing multi-lingual bibliometric data, and analyzing publication/patent abstracts at scale.

## Key Engineering Features & Methodology

### 1. Automated Data Ingestion & Formatting
* Establishes secure cloud infrastructure authentication using `google.colab.auth`.
* Engineers automated data streams using `gspread` to process raw textual strings (`processed_REF`) and structural target indices directly from Google Sheets into optimized `pandas` DataFrames.

### 2. Feature Engineering & Word Embeddings
To capture dense semantic meaning from text, the pipeline implements and compares two distinct embedding strategies:
* **Static Word Vectors (GloVe):** Ingests and parses Stanford's **Global Vectors for Word Representation** across multiple hyper-parameter dimensions (50d, 100d, 200d, and 300d). Implements a custom, object-oriented `GloveVectorizer` class extending Scikit-Learn's `BaseEstimator` and `TransformerMixin` to compute averaged document-level embeddings.
* **Contextual Deep Representations (ELMo):** Deploys **Embeddings from Language Models** via TensorFlow Hub (`hub.KerasLayer`), leveraging a deep, bi-directional language model (BiLSTM) to capture contextual, character-based word representations from text batches.

### 3. Statistical Modeling & Predictive Analytics
* **Robust Pipeline Construction:** Utilizes Scikit-Learn's `make_pipeline` to seamlessly link the feature vectorizers with mathematical classification heads.
* **High-Dimensional Classification:** Deploys multinomial Logistic Regression optimized via the **SAGA solver** to effectively handle high-dimensional, sparse embedding spaces across both **L1 (Lasso)** and **L2 (Ridge)** regularizations.
* **Validation & Error Analysis:** Guarantees zero data-leakage by executing a rigorous **10-Fold Stratified Cross-Validation** strategy (`StratifiedKFold`). The pipeline tracks and exports a granular matrix of performance metrics, including global Weighted Accuracy, global Mean Squared Error (MSE), and independent per-class error tracking.

## Repository Structure
```text
├── text_embedding_classification.py   # Production-ready Python script version of the pipeline
└── README.md                          # Project documentation and engineering summary
