# 🧠 Plagiarism Detection using NLP & Machine Learning

This project implements a **plagiarism detection system** using advanced **Natural Language Processing (NLP)** and **machine learning techniques**. It identifies both direct copying and subtle paraphrasing in textual data.

---

## 🚀 Overview

The system combines:
- 🧾 **Word Embeddings (GloVe)** for semantic understanding  
- 📊 **K-Means Clustering** to group similar words  
- 🔍 **FP-Growth Algorithm** to detect frequent patterns  

This hybrid approach enables accurate detection of plagiarism beyond simple text matching.

---

## 🎯 Objective

To develop a robust system capable of detecting:
- Exact plagiarism (copy-paste)
- Paraphrased content
- Semantic similarities across documents :contentReference[oaicite:0]{index=0}  

---

## 📂 Dataset

- Uses **Clough-Stevenson corpus**
- Contains:
  - 5 questions  
  - 20 answers per question  
  - Categories: `cut`, `heavy`, `light`, `non`  

- Average answer length: ~216 words :contentReference[oaicite:1]{index=1}  

---

## ⚙️ Methodology

### 1. Preprocessing
- Lowercasing text  
- Removing punctuation  
- Sentence tokenization using NLTK  
- Encoding handled with `latin1`  

### 2. Word Embeddings (GloVe)
- Converts words into vector representations  
- Captures semantic relationships  

### 3. K-Means Clustering
- Groups similar words into clusters  
- Helps in identifying semantic similarity  

### 4. Vector Representation
- Sentences converted into cluster-based vectors  
- Documents represented as transactions  

### 5. FP-Growth Algorithm
- Finds frequent patterns in transactions  
- Detects repeated sentence structures (plagiarism signals)  

---

## 🧪 Implementation Details

- Clustering using `sklearn.KMeans`  
- Pattern mining using `pyfpgrowth`  
- Text processing using `nltk` and `re`  
- Data handling using `pandas` and `numpy` :contentReference[oaicite:2]{index=2}  

---

## 📊 Results

- ✅ Precision: **85%**  
- ✅ Accuracy: **64%**  

Shows strong performance in detecting plagiarized content. :contentReference[oaicite:3]{index=3}  

---

## ▶️ Run the Project

Run the following command:

```bash
pip install numpy pandas nltk scikit-learn pyfpgrowth && python main.py
