# 🧠 Face Authentication System (Siamese Neural Network)

This project implements a **Face Verification System** using a **Siamese Neural Network** — a deep learning architecture that learns to measure similarity between two images.  
It demonstrates a complete end-to-end workflow for **training, evaluating, and performing large-scale inference** with both **metric learning** and **binary classification** approaches.

---

## 🚀 Project Overview

The goal of this project is to determine whether two face images belong to the same person.  
Two training paradigms are implemented:

1. **Metric Learning (Contrastive Loss)** — learns an embedding space where similar faces are close and dissimilar faces are far apart.  
2. **Binary Classification (Binary Crossentropy)** — learns to directly classify image pairs as *same* or *different*.

**Model trained on:**  
- **40,000 image pairs** (training)  
- **166,800 image pairs** (inference/test)  

**Framework:** TensorFlow 2.x  
**Training pipeline:** GPU-accelerated `tf.data` with lazy loading and efficient preprocessing.

---

## 📊 Results Summary

| Metric | Metric-Learning Model |
|--------|-----------------------|
| Train/Val Loss | **0.16** |
| Test Accuracy | **75%** |
| Precision | **74%** |
| Recall | **80%** |

**Confusion Matrix**

[[54860 24740]
[17323 69877]]

### Interpretation:  
- The model shows **strong recall (80%)**, meaning it successfully identifies most same-person pairs.  
- Accuracy is consistent at **~75%**, and the training loss converged stably at **0.16** for both train and validation.  
- The result is realistic for a baseline Siamese system trained from scratch without pretraining or augmentation.

---

## 🎯 Project Concept

This project is designed as a **comprehensive educational demonstration** of how real-world face authentication systems are built and evaluated.  
It showcases **end-to-end machine learning engineering**:
- Efficient data loading with `tf.data` and Python generators (lazy loading)
- GPU-based preprocessing
- Modular architecture with clean code structure
- Handling large inference datasets (166k+ pairs)
- Computing complete evaluation metrics (accuracy, confusion matrix, precision, recall)
- Adaptive thresholding for metric-learning systems

Rather than optimizing for the highest possible accuracy, the purpose is to **demonstrate the full ML workflow** — from dataset design to large-scale inference — in a way that reflects **real production pipelines**.

---

## 🧩 Project Structure

```
    Project Root/
│
├── src/ # Source code
│ ├── init.py
│ ├── config.py # Configuration settings
│ ├── data_loader.py # Data loading and preprocessing
│ ├── model.py # Model definitions
│ ├── utils.py # Utility functions
│ ├── train.py # Training scripts
│ └── inference.py # Inference scripts
│
├── models/ # Saved or pre-trained models
├── assets/ # Images, plots, or other assets
├── data/ # Datasets or processed data
├── .gitignore # Git ignore file
├── requirements.txt # Python dependencies
├── LICENSE # MIT License
└── README.md # Project documentation
```
---

## ⚙️ Key Features

- Two complete training paradigms:
  - **Metric Learning** with Contrastive Loss
  - **Binary Classification** with Binary Crossentropy
- **Lazy loading** and **GPU-based data preprocessing** using `tf.data.Dataset`
- **Efficient inference** on 166k test samples with TensorFlow pipelines
- **Adaptive thresholding** for the contrastive model
- **Comprehensive evaluation**: accuracy, precision, recall, and confusion matrix
- **Modular codebase** with reusable components for real-world ML pipelines

---

## 🧠 Model Architectures

### 1. Metric Learning (Contrastive Loss)
Each image passes through a shared **embedding extractor** network.  
The model learns to minimize the Euclidean distance between embeddings of same-person pairs and maximize it for different-person pairs.

### 2. Classification (Binary Crossentropy)
The embeddings are combined and passed through a **dense sigmoid output**, learning to predict directly whether two faces belong to the same person.

---

## Custom Components

This project includes custom implementations to enhance model training and flexibility:

### 1. Custom Contrastive Loss

A `Loss` class is implemented to define a **custom contrastive loss function**, which measures similarity between embeddings and helps the model learn discriminative features effectively.

```python
# Example usage
loss = ContrastiveLoss(margin=1.0)
```
### 2.Euclidean Distance Function
A euclidean_distance function is defined to compute the distance between two embedding vectors. This is used within the contrastive learning framework to calculate similarity scores.
```
    distance = euclidean_distance(embedding_a, embedding_b)
```

### 3.Lambda Layer Integration
The distance function is integrated into the model using a Lambda layer. This allows seamless end-to-end training within the Keras / TensorFlow model pipeline.
```
    from tensorflow.keras.layers import Lambda

    distance_layer = Lambda(euclidean_distance)([embedding_a, embedding_b])
    model.add(distance_layer)
```

These custom components enable the model to learn embeddings effectively, which is useful for tasks such as similarity matching, metric learning, and contrastive representation learning.

---

## 🔍 Inference & Evaluation

Inference uses the same TensorFlow data pipeline for efficient GPU processing:
```python
for imgA_batch, imgB_batch, labels_batch in dataset:
    distance = siamese_model([imgA_batch, imgB_batch], training=False)
    preds_batch = tf.cast(distance < threshold, tf.int32)
```

### The script computes:
- Accuracy
- Confusion Matrix
- Precision / Recall
- Adaptive threshold tuning for the metric-learning version


## 🧠 Adaptive Thresholding

For the contrastive model, predictions are made by comparing distances to a threshold t:
```
    y_pred = (distances < t).astype(int)
```

A simple sweep finds the optimal threshold:

```
    thresh_lst = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for t in thresh_lst:
        acc = accuracy_score(y_val, (distances < t))
```
This ensures the best trade-off between precision and recall.

## 📦 Requirements

- Python 3.8+
- TensorFlow 2.x (with GPU support)
- NumPy, Pandas, scikit-learn, Matplotlib

Install dependencies:
```
    pip install -r requirements.txt
```

## 🔬 Ways to Improve Accuracy

While this project is designed as an educational baseline, there are many techniques that can significantly boost performance:

| **Category**        | **Technique**                                                                                           | **Effect**                                                |
|----------------------|----------------------------------------------------------------------------------------------------------|-----------------------------------------------------------|
| **Data**             | Data augmentation *(rotation, flip, zoom, brightness, crop)*                                            | Improves generalization and robustness                    |
| **Architecture**     | Transfer learning using pretrained CNNs *(VGG16, ResNet50, MobileNetV2, EfficientNet)* as embedding extractor | Provides strong visual feature representations and faster convergence |
| **Loss Functions**   | Triplet Loss or ArcFace Loss                                                                            | Better separation in embedding space                      |
| **Regularization**   | Dropout, L2 regularization, Batch Normalization                                                         | Reduces overfitting                                       |
| **Training Strategy**| Hard Negative Mining


## 🧠 Educational Focus
This project is primarily built as a training and demonstration project for:
- Understanding Siamese architectures and metric learning
- Learning efficient TensorFlow data pipelines
- Practicing clean, modular ML engineering
- Performing realistic large-scale inference and evaluation

The structure reflects how real MLOps-oriented research and production prototypes are organized.

## 🧑‍💻 Author
Armin Golzar
AI Engineer | Deep Learning • Computer Vision 
📍 GitHub: armingolzar
📧 Email: armingolzar78@gmail.com
