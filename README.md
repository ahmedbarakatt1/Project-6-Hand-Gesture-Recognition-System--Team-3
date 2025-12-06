# ✋ Hand Gesture Recognition System  
**Team 3 — Project 6**  
🎓 Graduation Project — Digital Egypt Pioneers Initiative (DEPI) / Microsoft Machine Learning Program  

A deep-learning powered system that recognizes hand gestures from images and real-time video. Built using computer vision, convolutional neural networks (CNNs), and OpenCV, this project enables gesture-based interaction for applications such as HCI, VR, gaming, and accessibility tools.

---

## 📄 Table of Contents
- [Overview](#overview)  
- [DEPI Graduation Project Statement](#depi-graduation-project-statement)  
- [Features](#features)  
- [Model Architecture](#model-architecture)  
- [Project Structure](#project-structure)  
- [Installation](#installation)  
- [Dataset & Preprocessing](#dataset--preprocessing)  
- [Training & Evaluation](#training--evaluation)  
- [Real-Time Gesture Recognition](#real-time-gesture-recognition)  
- [Results](#results)  
- [Team](#team)  

---

## 👀 Overview  
The Hand Gesture Recognition System detects and classifies hand gestures using a trained deep-learning model.  
This work includes:  
- Dataset preprocessing & augmentation  
- CNN model training & evaluation  
- Real-time webcam-based gesture detection  
- Deployment-ready modules — enabling further integration  

---

## 🎓 DEPI Graduation Project Statement  
This repository contains the graduation project of DEPI for the Microsoft Machine Learning Program. It represents our final applied project, showcasing practical machine learning skills in computer vision, deep learning, and real-time AI deployment. :contentReference[oaicite:2]{index=2}

---

## ✨ Features  
-  Preprocessing pipeline for gesture images (resizing, normalization, augmentation)  
-  Data augmentation to improve model generalization  
-  CNN-based gesture classification (custom CNN and support for transfer learning)  
-  Real-time hand gesture detection using webcam + OpenCV  
-  Optional integration with UI / deployment modules (for real-world use)  
-  Model evaluation (accuracy, precision, recall, F1-score, confusion matrix)  

---

## ✋ Model Architecture  
The system supports custom CNN models (tailored to the dataset and gestures)  

---

## 👨‍🏫  Project Structure  

```text
/
├── Final Model/               ← Saved trained models  
├── Notebooks/                 ← Jupyter notebooks (preprocessing, training, evaluation)  
├── Preprocessed Data/         ← Cleaned & augmented datasets  
├── Presentations/             ← Project slides & demo material  
├── Reports/                   ← Final reports, documentation
├── Streamlit UI App/          ← Web-app for real-time gesture recognition    
├── README.md                  ← This file  
└── LICENSE                    ← MIT License  
```

---

## ⚙️ Installation

To run this project locally, follow these steps:

```bash
# Clone the repository
git clone https://github.com/ahmedbarakatt1/Project-6-Hand-Gesture-Recognition-System--Team-3.git
cd Project-6-Hand-Gesture-Recognition-System--Team-3
```

# (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# To launch the Streamlit UI for real-time gesture recognition:

```bash
streamlit run "Streamlit UI App/app.py"
```

---

## 🗂️ Dataset & Preprocessing

The dataset used in this project is the Kaggle hand gesture recognition dataset, composed by a set of near infrared images acquired by the Leap Motion sensor.
To ensure high-quality model training, the following preprocessing steps were applied:

- **Image resizing** for consistent input dimensions  
- **Normalization** to stabilize and speed up training  
- **Background removal** (optional) to reduce noise  
- **Data augmentation**, including:  
  - Rotation  
  - Flipping  
  - Scaling  
  - Brightness and contrast adjustments  

All processed and augmented data is stored inside the **Preprocessed Data/** directory to maintain reproducibility and allow consistent training across environments.

---

## 🧠 Training & Evaluation

Model training and evaluation are performed using Jupyter notebooks available in the **Notebooks/** directory.

This phase includes:

- Building and training **custom CNN** or **transfer-learning** models  
- Splitting the data into **training**, **validation**, and **testing** sets  
- Tracking key evaluation metrics:  
  - Accuracy  
  - Precision  
  - Recall  
  - F1-score  
  - Confusion matrix  
- Visualizing training progress using:  
  - Accuracy curves  
  - Loss curves  
- Saving trained models to the **Final Model/** directory for reusability and deployment  

---

## 🎥 Real-Time Gesture Recognition

The real-time gesture recognition module enables live detection using a webcam feed.  

**Real-time processing includes:**

- **OpenCV** for webcam capture  
- **Preprocessing** applied to each incoming frame  
- **Model inference** to classify the gesture  
- **On-screen overlays** showing the predicted class  
- **Streamlit UI** for an interactive and user-friendly experience  

To run the live demo:

```bash
streamlit run "Streamlit UI App/app.py"
```

---

## 📊 Results

The project demonstrates strong performance in static hand-gesture recognition.  
Several evaluation visualizations are included in the notebooks, such as:

- **Training vs. validation accuracy curves**
- **Training vs. validation loss curves**
- **Confusion matrix** 

The real-time detection system performs reliably under typical lighting conditions and standard webcam setups, providing smooth and accurate gesture predictions during live testing.

---

## 👥 Team

*   Aya Ashraf — [LinkedIn](https://www.linkedin.com/in/aya-ashraf15?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=ios_app)
*   Tasnim Qutb — [LinkedIn](https://www.linkedin.com/in/tasnim-qotb?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app)
*   Ahmed Barakat — [LinkedIn](https://www.linkedin.com/in/ahmad-barakat-4a16101b9?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app)
*   Mohammed Osama — [LinkedIn](https://www.linkedin.com/in/mohamed-abdulhalem?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=ios_app)
*   Mohammed Saied — [LinkedIn](https://www.linkedin.com/in/mohammed-sa3ied?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app)

---

## 🙏 Acknowledgments
**Special thanks to:**
* DEPI & MCIT for mentorship and support
* Microsoft Machine Learning Engineer Program instructors
* Our colleagues and fellow trainees

---

## 📝 License
This project is licensed under the **MIT License**.

You may use, modify, and distribute this project under its terms.
