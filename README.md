# Text-and-Image-classification
#  Learning to Label: Clustering-Driven Text and Image Classification

##  Overview

This project builds an end-to-end machine learning pipeline to **auto-generate pseudo-labels** from unlabeled data using clustering, followed by training **supervised classifiers** on those labels.

### Domains Covered:
- **Text Classification** (NLP)
- **Image Classification** (Computer Vision)

---

##  Objective

Design and implement a pipeline that:
- Uses **unsupervised clustering** to assign pseudo-labels
- Applies **supervised learning models** trained on these pseudo-labels
- Benchmarks model performance on both text and image datasets

---

##  Pipeline Summary

| Stage                     | Techniques Used                                                  |
|--------------------------|------------------------------------------------------------------|
| Feature Engineering       | HOG (Images), Word2Vec, TF-IDF (Text)                           |
| Dimensionality Reduction  | PCA (Principal Component Analysis)                              |
| Unsupervised Clustering   | KMeans                                                          |
| Visualization             | t-SNE                                                           |
| Classification Models     | Logistic Regression, KNN, SVM, Random Forest, Neural Networks   |

---

##  Clustering-Driven Text Classification

## Dataset
- Unlabeled textual data
- Tools used: `pandas`, `pickle`, `nltk`, `BeautifulSoup`

## Preprocessing
- Lowercasing, punctuation/special character removal, HTML tag removal
- Tokenization, stopword removal (using NLTK)
- Lemmatization and stemming

## Feature Extraction
- TF-IDF
- Word2Vec (CBOW and Skip-gram)
- Chose **CBOW** based on superior cosine similarity

## Clustering
- Applied **PCA** to reduce vector space
- Used **KMeans** to generate cluster-based pseudo-labels

## Classification Results

| Model                | Accuracy |
|---------------------|----------|
| Logistic Regression | 91%      |
| **K-Nearest Neighbors** | **98%** ( Chosen) |
| Random Forest        | 97%      |

---

##  Clustering-Driven Image Classification

## Dataset
- Shape: (60000, 28, 28) → Flattened to (60000, 784)
- Normalized to [0, 1]
- Libraries: `numpy`, `pandas`

## Preprocessing
- Removed channel dimension
- Flattened images
- Normalized pixel values

## Feature Engineering
- Used **HOG (Histogram of Oriented Gradients)** to extract texture/edge features
- Converted each image into a 144-dimension feature vector

## Dimensionality Reduction
- Applied PCA to reduce HOG features to 100 dimensions

## Clustering
- Applied **KMeans** to PCA-reduced features
- Output: 10 clusters (digits 0–9)

## Visualization
- Applied **t-SNE** on PCA vectors for visual inspection
- Found that clusters 0, 1, 3, 6, 9 showed good separation

## Classification Results

| Model                     | Accuracy |
|--------------------------|----------|
| SVM (linear + RBF)       | ~92%     |
| Random Forest (best config) | **95%** |
| Neural Network (5 neurons) | **94%** ( Chosen) |

---

##  Libraries Used

```text
numpy
pandas
scikit-learn
nltk
matplotlib
seaborn
tensorflow
keras
BeautifulSoup
