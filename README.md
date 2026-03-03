# Bone Fracture Detection using Deep Learning

## Overview
A deep learning model to detect bone fractures from X-ray images using ResNet50 transfer learning.

## Results
| Metric | Score |
|--------|-------|
| Test Accuracy | 97.37% |
| ROC AUC | 0.9970 |
| Fracture F1 | 0.98 |
| Normal F1 | 0.94 |

## Dataset
- Total images: 2127 (Fracture: 2000, Normal: 127)
- After augmentation: 1900 train, 380 val, 380 test

## Model
- Architecture: ResNet50 (pretrained on ImageNet)
- Trainable layers: Layer3, Layer4, FC
- Dropout: 0.4
- Optimizer: Adam (lr=0.0001, weight_decay=1e-4)

## Project Structure
```
BoneFractureProject/
├── model/         # Trained model weights
├── outputs/       # Plots and visualizations  
├── notebooks/     # Colab training notebook
├── app/           # FastAPI and Gradio apps (coming soon)
└── requirements.txt
```

## Libraries
- PyTorch, torchvision
- scikit-learn, matplotlib, seaborn
- grad-cam, Gradio, FastAPI

## Model Weights
Model file stored in Google Drive (too large for GitHub).
