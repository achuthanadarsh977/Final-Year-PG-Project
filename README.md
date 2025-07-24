# Silhouette-Based Gait Recognition Using Local Temporal Aggregation

## 🧑‍💻 Submitted by  
**Adarsh Achuthan**  
MCA (2023178015)  
Department of Information Science and Technology  
College of Engineering, Guindy, Anna University  
April 2025

---

## 📄 Abstract

This project explores gait recognition, a biometric authentication technique that identifies individuals based on walking patterns. Unlike fingerprint or iris scans, gait can be captured unobtrusively using video. The proposed method uses silhouette frames extracted from gait videos and enhances them using local temporal aggregation and global-local feature extraction. The system is tested on the CASIA-B dataset and shows improved recognition accuracy and robustness.

---

## 🎯 Objectives

- Extract silhouette frames from CASIA-B dataset videos  
- Segment and analyze human gait using spatial shape and motion dynamics  
- Combine features for robust and accurate gait recognition  
- Improve performance using cross-entropy and triplet loss functions  

---

## 🧠 System Overview

- **Data Acquisition**: CASIA-B dataset (124 subjects, 11 views, 3 conditions)
- **Feature Extraction**:
  - Local Temporal Aggregation (LTA)
  - Global and Local Feature Extraction (GLFE)
- **Temporal Pooling**: Generalized Mean (GeM) pooling
- **Classification**: Fully Connected (FC) layers with Batch Normalization
- **Loss Calculation**: Cross-Entropy + Triplet Loss

---

## 🛠️ Implementation Details

- **Preprocessing**:
  - Grayscale conversion, background subtraction, normalization
- **Model Architecture**:
  - Two Conv2D + MaxPooling layers
  - Dense layer with ReLU
  - Softmax output layer
- **Training**:
  - Optimized using Adam optimizer
  - Batch All strategy for triplet loss
  - Training accuracy: >99%
- **Testing**:
  - Rank-1 accuracy: 83.6% on unseen gait samples

---

## 🧪 Results and Analysis

| Metric                  | Value         |
|-------------------------|---------------|
| Training Accuracy       | >99%          |
| Validation Accuracy     | ~20% (overfitting noted) |
| Test Accuracy (Rank-1)  | 83.6%         |
| Confidence              | >90% for most test predictions |

---

## 📈 Future Work

- Add data augmentation to improve generalization  
- Use deeper or more regularized CNN architectures  
- Explore real-time deployment strategies  
- Apply to broader surveillance and healthcare contexts  

---



