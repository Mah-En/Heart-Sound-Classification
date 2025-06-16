# Heart-Sound-Classification


## **1. Introduction**

We want to recognize different types of heart sounds using the **Pascal Heart Sound Dataset**.  
There are **5** sound types:

1. **Normal**  
2. **Murmur**  
3. **Extra Heart Sound**  
4. **Artifact**  
5. **Extrasystole**

This dataset is made by combining:

- **Dataset A**: Public recordings from iStethoscope Pro  
- **Dataset B**: Hospital recordings from DigiScope  

Goal is to build a model that classifies each recording into one of these five categories.

---

## **2. Preprocessing**

1. **Denoising**  
   - Remove background noise to make heart sounds clearer (e.g., wavelet denoising, spectral gating).

2. **Resampling**  
   - I unify all recordings to the same sampling rate (e.g., 4000 Hz).

3. **Segmentation & Padding**  
   - Fix the length of each clip, for example, 4 seconds. If a clip is shorter, add zeros (padding).

4. **Feature Extraction (MFCCs)**  
   - Convert audio into **Mel-Frequency Cepstral Coefficients (MFCCs)**, which capture important sound features in a 2D format.

---

## **3. Model Architecture**

### 3.1 **CNN + RNN Framework**

1. **CNN (Convolutional Neural Network)**  
   - Learns short-term time–frequency patterns in the MFCCs, treating them like images.

2. **RNN (Recurrent Neural Network)**  
   - Focuses on how these patterns evolve over time. We try different RNN cells:

   - **Simple RNN**: Basic approach.  
   - **LSTM**: Better at keeping long-term memory.  
   - **Bi-LSTM**: Looks at signals forward and backward.  
   - **GRU**: Similar to LSTM but fewer parameters.  
   - **xLSTM**: An extended/custom LSTM concept.

### 3.2 **Attention Mechanism **

- **Attention**: Allows the model to zoom in on the most important parts of the heartbeat.  
- LSTM with attention often boosts performance and helps us understand which parts of the audio matter most.

---

## **4. Training Details**

- **Loss Function**: CrossEntropy (for multiple classes).  
- **Optimizer**: Adam (learning rate around \(1\times10^{-3}\)).  
- **Data Split**: 70% Train, 15% Validation, 15% Test (or a similar ratio).  
- **Epochs**: Usually 10–20, checking validation accuracy to stop early if needed.

---

## **5. Results**

Measure:

- **Accuracy**  
- **Precision, Recall, F1-Score**  
- **Confusion Matrix**  

---

## **6. Discussion**

1. **Simple RNN**: Struggles with longer audio sequences.  
2. **LSTM & GRU**: Capture longer relationships better, leading to improved scores.  
3. **Bi-LSTM**: Gains context by processing sounds forward and backward.  
4. **Attention**: Helps the model focus on key heartbeats, improving both accuracy and interpretability.