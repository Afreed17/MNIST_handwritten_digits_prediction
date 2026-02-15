# 🧠 MNIST Handwritten Digit Recognition

This project implements multiple Machine Learning and Deep Learning models to classify handwritten digits (0–9) using the MNIST dataset. The objective is to compare classical ML algorithms with deep learning approaches and identify the best-performing model for production use.

---

## 📌 Project Objective

- Perform complete data analysis on the MNIST dataset.
- Build classification models for handwritten digit recognition.
- Compare multiple models and identify the best classifier.
- Provide a performance comparison report with limitations and challenges.

---

## 📊 Dataset Information

- Dataset: MNIST (Modified National Institute of Standards and Technology)
- Training Samples: 60,000
- Testing Samples: 10,000
- Image Size: 28 × 28 grayscale
- Number of Classes: 10 (digits 0–9)

MNIST is a benchmark dataset widely used for evaluating classification algorithms in computer vision.

---

## 🛠️ Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- TensorFlow / Keras
- Matplotlib

---

## 🤖 Models Implemented

### 1️⃣ Logistic Regression
- Baseline linear classifier
- Accuracy: ~92–93%
- Fast training and inference

### 2️⃣ Linear SVM
- Margin-based classifier
- Accuracy: ~91–93%
- Efficient for high-dimensional data

### 3️⃣ RBF Kernel SVM
- Non-linear classifier
- Accuracy: ~97–98%
- Computationally expensive

### 4️⃣ Convolutional Neural Network (CNN)
- Deep learning model optimized for images
- Preserves spatial structure
- Accuracy: ~98–99%
- Best performing model

---

## 📈 Model Comparison

| Model | Accuracy | Speed | Complexity |
|--------|----------|--------|------------|
| Logistic Regression | ~92% | Very Fast | Low |
| Linear SVM | ~92% | Fast | Medium |
| RBF SVM | ~98% | Slow | High |
| CNN | ~98–99% | Moderate | High |

---

## 🏆 Best Model

The **Convolutional Neural Network (CNN)** achieved the highest accuracy and demonstrated superior generalization. It is recommended for real-world deployment.

---

## ⚠️ Limitations

- MNIST is a clean dataset; real-world handwriting may be noisier.
- CNN requires more computational resources.
- Limited hyperparameter tuning performed.
- No advanced data augmentation implemented.

---

## 🚧 Challenges Faced

- Correct data reshaping and normalization
- Choosing appropriate activation functions (Softmax vs Sigmoid)
- Matching correct loss functions
- Avoiding overfitting in neural networks
- Managing computational cost of RBF SVM

---

## 🚀 Future Improvements

- Hyperparameter tuning using GridSearch / RandomSearch
- Data augmentation for robustness
- Early stopping and learning rate scheduling
- Deployment using TensorFlow Lite
- Testing on real-world handwritten datasets

---



