# 🛢️ Oil Production Anomaly Detection (LSTM Autoencoder)

## 📌 Overview
This project implements a deep learning–based anomaly detection system for oil well production data using an **LSTM Autoencoder**.  
By modeling normal operating behavior as a multivariate time-series, the autoencoder learns the “healthy” production patterns.  
When the reconstruction error is unusually high, the corresponding points are flagged as **anomalies**, which may indicate:

- Equipment malfunction  
- Operational instability  
- Sensor/data quality issues  

All core logic—data processing, model construction, training, evaluation, and visualization—lives inside the **`oil_anomaly.ipynb`** Jupyter Notebook.

---

## 🎯 Goal
To accurately detect abnormal daily events in the oil production rate of well **NO 15/9-F-14 H** using key multivariate time-series features such as:

- Oil Volume  
- Gas Volume  
- Hours on Stream  
- And engineered metrics (e.g., Oil Production Rate)

---

## 🛠️ Methodology

### **1. Data Preparation**
- Filter the dataset for the target well.  
- Compute **Oil Production Rate (BBL/HR)**.  
- Handle zero-division cases when `ON_STREAM_HRS = 0`.  
- Address missing values caused by shutdowns or sensor gaps.

### **2. Feature Scaling**
- Apply **MinMaxScaler** trained *only on the training subset* to avoid data leakage.

### **3. Sequence Generation**
- Convert the time-series into **30-step sequences**:  
[num_samples, 30, num_features]


### **4. LSTM Autoencoder**
A sequence-to-sequence model is built:

- **Encoder:** compresses input sequences into latent representation  
- **Decoder:** reconstructs the original sequence  
- Learned representation captures “normal behavior” of the well

### **5. Anomaly Threshold**
- Compute the **Mean Absolute Error (MAE)** on training reconstruction.  
- Set anomaly threshold at the **99th percentile** of training MAE.  
- Any test sample exceeding this value = **anomaly**.

---

## How to Run the Project

### **Prerequisites**
Install required dependencies:
```bash
pip install pandas numpy matplotlib scikit-learn tensorflow keras missingno joblib
```

### Steps
#### 1. Clone Repository
- git clone (https://github.com/congthinh2132/Oil-Anomoly-Detection.git)
- cd Oil-Anomoly-Detection

#### 2. Add Dataset
- Place your raw production dataset in the project root.

#### 3. Run the Notebook
- Open oil_anomaly.ipynb in JupyterLab or VS Code and execute all cells.
- The notebook will:
  - Load & preprocess raw data
  - Train the LSTM Autoencoder (with EarlyStopping)
  - Compute reconstruction errors
  - Derive anomaly threshold
  - Produce the final anomaly visualization plot
  - Comparing between timing window 60 days and 90 days

## Results Summary
- The model detected 80 anomalous days in the test period. (60 days time slicing)
- The anomalies strongly correspond to periods where the well shows:
  - Sudden drops in oil rate
  - Sharply fluctuating production
  - Unusual operating behavior

These findings demonstrate the model’s effectiveness in identifying operational irregularities that merit further engineering review.

<img width="1242" height="547" alt="image" src="https://github.com/user-attachments/assets/835ad2bb-323e-4163-af65-d0c8ab4ec20e" />

<img width="1229" height="547" alt="image" src="https://github.com/user-attachments/assets/ae9e7612-c363-4e4c-93ae-aa374494a5bf" />

