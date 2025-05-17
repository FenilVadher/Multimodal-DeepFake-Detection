# 🤖 Multimodal DeepFake Detection

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Online-blue)](https://multimodal-deepfake-detection-p04k.onrender.com)

Multimodal DeepFake Detection is an AI-powered system designed to detect deepfake content by analyzing **audio**, **video**, and **text** simultaneously. It leverages advanced machine learning and deep learning models to provide robust and accurate fake media detection by identifying inconsistencies across multiple data modalities.

---

## 🧠 Features

- 🎙️ **Audio Analysis**: Detects tampered audio using MFCC features and traditional ML models.
- 📷 **Image/Video Analysis**: Uses CNN (ResNet18) to analyze visual inconsistencies in video frames.
- 📝 **Textual Analysis**: Employs BERT embeddings and logistic regression to detect AI-generated text.
- 🧩 **Multimodal Fusion**: Combines predictions from all three sources for final verdict.
- 🌐 **Interactive Web App**: Built using Streamlit for intuitive user interaction.

---

## 🚀 Live Demo

🔗 [Click here to view the deployed project](https://multimodal-deepfake-detection-p04k.onrender.com)

---

## 🛠️ Installation

Follow the steps below to set up and run the project locally:

### 1. Clone the repository

```bash
git clone https://github.com/FenilVadher/Multimodal-DeepFake-Detection.git
cd Multimodal-DeepFake-Detection

# Create a virtual environment (optional)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt

# Run the application
streamlit run DeepFakeDetection.py

