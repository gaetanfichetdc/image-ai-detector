# AI vs Real Image Classifier

This project trains a deep learning model to distinguish between **AI-generated images** and **real images** using transfer learning.  
It is designed as a portfolio project to showcase skills in data preprocessing, model training, evaluation, and interpretability with Grad-CAM.

---

## 📌 Project Goals
- Build a binary classifier that predicts whether an image is **REAL** or **FAKE (AI-generated)**.  
- Use **transfer learning** with a pretrained ResNet-18 to achieve strong performance with limited training time.  
- Explore **interpretability** using Grad-CAM heatmaps to visualize what regions of an image influence the model’s decision.  
- Provide a reproducible notebook that others can run to reproduce the results.

---

## 📂 Dataset
The dataset used is **CIFAKE: Real and AI-Generated Synthetic Images**, available on Kaggle:  
👉 [CIFAKE dataset on Kaggle](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images?resource=download)  

- **REAL images**: taken from the CIFAR-10 dataset (natural images).  
- **FAKE images**: generated using a latent diffusion model to mimic CIFAR-10 classes.  
- The dataset is balanced, with ~60k real and ~60k synthetic images.

Folder structure after extraction:

train/
FAKE/
REAL/
test/
FAKE/
REAL/

## 🚀 How to Run

### 1. Open the notebook
The main notebook is **`ML.ipynb`**. Open it in Jupyter Notebook, VS Code, or Kaggle.

### 2. Install dependencies
This project uses PyTorch, torchvision, scikit-learn, matplotlib, and seaborn.  
You can install them with:
```bash
pip install torch torchvision scikit-learn matplotlib seaborn
```
