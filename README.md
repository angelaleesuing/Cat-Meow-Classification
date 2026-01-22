# Cat Meow Classification 🐱🔊

## SECB3203 Programming for Bioinformatics  
**Semester 1 2025/2026**  
**Section 02 – Group 6**

### Group Members 
- **Angela Lee Su Ing** (A23CS0047)  
- **Karen Yam Vei Xin** (A23CS0093)
- **Nurul Syasyawafa Binti Amran** (A23CS0167) 

**Lecturer:** Dr. Seah Choon Sen  
**Submission Date:** 22 January 2026  

---

## 📌 Project Overview
Cat vocalizations play an important role in communication between cats and humans. Different meows can indicate hunger, distress, attention-seeking behavior, or other emotional states. Traditionally, interpreting these sounds relies on human judgment, which can be subjective and inefficient when dealing with large datasets.

This project proposes an **automated machine learning-based system** to analyze and classify cat meow sounds using **audio feature extraction** and **statistical learning techniques**. The project focuses on applying bioinformatics and data science methods to biological audio signals.

---

## 🎯 Objectives
- To preprocess raw cat meow audio signals into numerical features  
- To extract **Mel-Frequency Cepstral Coefficients (MFCCs)** from audio recordings  
- To perform **Exploratory Data Analysis (EDA)** on extracted features  
- To develop regression-based machine learning models  
- To evaluate model performance using **R-squared (R²)** and **Mean Squared Error (MSE)**  

---

## 📂 Dataset
- **Source:** Kaggle – Cat Meow Classification Dataset  
- **Total Samples:** 440 audio recordings  
- **Breeds:** Maine Coon, European Shorthair  
- **Contexts:** Different emotional and behavioral states  
- **Format:** WAV audio files  

---

## ⚙️ Methodology

### 1. Data Collection & Pre-processing
- Audio files were imported using the **Librosa** library  
- Original sampling rates were preserved  
- MFCC features were extracted from each audio file  
- Extracted features were stored in a structured CSV format  

### 2. Data Wrangling
- Data organized into Pandas DataFrames  
- Checked for missing or invalid values (none found)  
- Feature normalization and scaling applied  
- Ensured numerical consistency across MFCC features  

### 3. Exploratory Data Analysis (EDA)
- Descriptive statistics (mean, standard deviation, min, max)  
- Correlation analysis of MFCC features  
- Visualizations such as correlation heatmaps and distribution plots  

### 4. Model Development
- Simple Linear Regression  
- Multiple Linear Regression  
- Polynomial Regression  
- Ridge Regression with regularization  

### 5. Model Evaluation
- Train-test split for performance evaluation  
- Metrics used:
  - R-squared (R²)
  - Mean Squared Error (MSE)  
- Visualization of actual vs predicted values  
- Regularization applied to reduce overfitting  

---

## 🧪 Testing & Validation
- Models tested on unseen data to simulate real-world usage  
- Ridge regression showed improved generalization  
- Hyperparameter tuning enhanced model stability  
- Results indicate meaningful relationships between MFCC features  

---

## 🛠️ Software & Hardware Requirements

### Software
- **Programming Language:** Python 3.9+  
- **IDE:** VS Code / Jupyter Notebook  
- **Version Control:** Git & GitHub  
- **Libraries:**  
  - NumPy  
  - Pandas  
  - Matplotlib  
  - Seaborn  
  - Scikit-learn  
  - Librosa  
  - TensorFlow / PyTorch (optional)  
- **Optional Cloud Platforms:** Google Colab, AWS, Azure  

### Hardware
- **CPU:** Intel i5 / Ryzen 5 (minimum)  
- **RAM:** 8 GB (16 GB recommended)  
- **Storage:** 10 GB free  
- **GPU (Optional):** NVIDIA GPU for deep learning  
- **OS:** Windows / macOS / Linux  

---

## 📊 Results
- Regression models successfully captured relationships between MFCC features  
- Ridge regression reduced overfitting and improved stability  
- The workflow demonstrates the feasibility of applying machine learning to biological audio analysis  

---

## 📌 Conclusion
This project demonstrates the application of machine learning techniques to classify and analyze cat vocalization sounds. By combining audio signal processing, exploratory data analysis, and regression modeling, the study highlights the potential of bioinformatics methods beyond traditional genomic data. The proposed system provides a foundation for future research in automated animal sound recognition.

---

## 📎 Notes
- This project focuses on **offline audio analysis**
- Real-time sound recognition and hardware deployment are outside the scope

