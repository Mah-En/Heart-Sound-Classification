# Heart Sound Classification: A Deep Learning Approach Using CNN-RNN Architectures with Attention

## Project Overview

This project focuses on the automated classification of heart sounds into distinct categories (Normal, Murmur, Extra Heart Sound, Artifact, Extrasystole) using deep learning. It leverages a hybrid Convolutional Neural Network (CNN) and Recurrent Neural Network (RNN) framework, incorporating advanced techniques like wavelet denoising, MFCC feature extraction, and a soft attention mechanism to enhance classification accuracy and interpretability.

## 1. Introduction

Cardiovascular diseases remain a leading cause of mortality globally. Early and accurate diagnosis, often aided by analyzing heart sounds (phonocardiography), is crucial. However, manual interpretation of these sounds requires significant medical expertise and can be time-consuming. This project aims to automate this diagnostic step using deep learning, making it more efficient and accessible.

The core task is to build a robust model that can classify heart sound recordings into five specific categories:
* **Normal**: Healthy heart sounds.
* **Murmur**: Sounds indicative of turbulent blood flow.
* **Extra Heart Sound (Extrahls)**: Additional, often pathological, sounds.
* **Artifact**: External noise or interference.
* **Extrasystole**: Premature heartbeats.

The project utilizes the diverse Pascal Heart Sound Dataset, combining public recordings from iStethoscope Pro (Dataset A) and hospital recordings from DigiScope (Dataset B).

## 2. Dataset Description

The project relies on two main datasets, aggregated into a unified collection of `.wav` audio files:

* **Pascal Heart Sound Dataset (Dataset A & B)**:
    * **Source**: A combination of public recordings (iStethoscope Pro) and hospital recordings (DigiScope).
    * **Categories**: Audio files are organized into folders corresponding to the five distinct labels: `normal`, `murmur`, `extrahls` (Extra Heart Sound), `artifact`, and `extrasystole`.
    * **Content**: Each file contains raw audio signals of heart sounds.

### Key Characteristics and Initial Observations

Understanding the raw audio data's characteristics is vital for effective preprocessing:

* **Varied Sampling Rates**: Recordings may have different original sampling rates, requiring standardization.
* **Variable Durations**: Audio clips vary significantly in length (from <1 second to >27 seconds), necessitating segmentation or padding for fixed model input.
* **Presence of Noise**: Medical recordings often contain background noise, demanding denoising techniques.
* **Class Imbalance**: A significant challenge, as 'Normal' sounds are the majority class, potentially biasing the model.

### Dataset Visualizations

#### Distribution of Audio Durations

This histogram shows the varying lengths of the heart sound recordings. Most recordings are relatively short, but some extend beyond 20 seconds.

![Distribution of Audio Durations](figures/output-3.png)
*Figure 1: Distribution of Audio Durations*

#### Distribution of Heart Sound Labels

This bar chart illustrates the class imbalance within the dataset, where 'normal' heart sounds are significantly more frequent than the other categories.

![Distribution of Heart Sound Labels](figures/output-4.png)
*Figure 2: Distribution of Heart Sound Labels*

## 3. Preprocessing and Feature Extraction

Raw audio signals are high-dimensional and often noisy. The following steps transform them into a clean, standardized, and informative format for deep learning:

### 3.1. Denoising

To remove background noise from the recordings, a wavelet-based denoising technique is applied:
* **Wavelet Decomposition**: The signal is decomposed into frequency sub-bands using the `db4` wavelet.
* **Thresholding**: A robust universal threshold (based on median absolute deviation) is used for soft thresholding, suppressing noise coefficients.
* **Wavelet Reconstruction**: The signal is reconstructed from the thresholded coefficients.

### 3.2. Resampling

All audio recordings are converted to a consistent sampling rate of 4000 Hz to ensure uniformity and reduce computational load.

### 3.3. Segmentation and Padding

To meet the fixed input dimension requirements of neural networks:
* All clips are standardized to a desired length (e.g., 4 seconds).
* Shorter clips are zero-padded at the end.
* Longer clips are truncated to the fixed length.

### 3.4. Feature Extraction: Mel-Frequency Cepstral Coefficients (MFCCs)

MFCCs are widely used in audio processing as they mimic human hearing and efficiently summarize sound characteristics.
* **MFCC Generation**: For each preprocessed audio signal, 40 MFCCs are computed using `librosa.feature.mfcc`.
* **2D Representation**: MFCCs provide a 2D representation (n_mfcc x time_frames), suitable for CNN inputs, capturing how the sound's frequency content changes over time.

### 3.5. `HeartSoundDataset` Class and `DataLoader`

A custom `HeartSoundDataset` class encapsulates all preprocessing and feature extraction steps. It also handles label encoding (mapping string labels to numerical indices). `torch.utils.data.DataLoader` is used to efficiently batch and shuffle these processed MFCC features and labels for model training.

## 4. Data Splitting

The dataset is split into training, validation, and testing sets to ensure robust model evaluation and prevent data leakage:

* **Stratified Split**: `train_test_split` with `stratify` on the `label` column is used to maintain original class proportions in all splits, crucial due to the severe class imbalance.
* **Ratios**: The data is split approximately into 60\% training, 20\% validation, and 20\% testing.
    * **TRAIN size**: 351 samples
    * **VAL size**: 117 samples
    * **TEST size**: 117 samples

## 5. Model Architectures

The project employs a hybrid CNN-RNN architecture to leverage the strengths of both network types:

* **CNN Feature Extractor**: Processes 2D MFCC spectrograms to extract spatial (time-frequency) features. It consists of two convolutional layers with batch normalization, ReLU activation, and max pooling, transforming raw MFCCs into a compact feature representation.

* **RNN Classifiers**: Reshaped CNN outputs (sequences of features over time) are fed into various RNN architectures to model temporal dependencies:

    * **Simple RNN Classifier**: A basic RNN layer. It often struggles with long-term dependencies.
    * **LSTM Classifier**: Utilizes `nn.LSTM` with gate mechanisms to better capture long-term dependencies. Supports `bidirectional` processing.
    * **Bi-LSTM (Bidirectional LSTM)**: Processes sequences in both forward and backward directions, concatenating outputs for richer contextual understanding.
    * **GRU Classifier**: A simplified LSTM (`nn.GRU`) with fewer parameters, often achieving comparable performance.
    * **xLSTM Classifier (Conceptual Placeholder)**: Implemented as a standard LSTM in this project; a full xLSTM would integrate more advanced features (e.g., attention, residual connections, novel gating) for enhanced capabilities.

* **Attention Mechanism**: A soft attention layer is integrated into the LSTM classifier (`LSTMAttentionClassifier`). This mechanism dynamically assigns weights to different time segments of the MFCC sequence, allowing the model to focus on the most salient parts of the heart sound for classification. This improves both accuracy and offers a degree of interpretability.

## 6. Training and Evaluation Utilities

Standardized functions ensure a structured training and evaluation process:

* **`train_model` Function**:
    * Moves models to the appropriate device (CPU/GPU).
    * Uses `nn.CrossEntropyLoss` for multi-class classification.
    * Employs the `Adam` optimizer with an initial learning rate of `1e-3`.
    * Manages the training loop, including forward pass, loss calculation, backpropagation, and optimizer steps.
    * Performs validation after each epoch, computing validation accuracy.

* **`evaluate_model` Function**:
    * Sets the model to evaluation mode (`model.eval()`).
    * Collects predictions and true labels without gradient computation.
    * Prints overall **Accuracy**, a detailed **Classification Report** (precision, recall, F1-score for each class), and a **Confusion Matrix**.

## 7. Results

The various CNN-RNN hybrid models were trained for 10 epochs. Their performance on the unseen test set is summarized below:

### Overall Test Accuracy

| Model                       | Test Accuracy |
| :-------------------------- | :------------ |
| CNN + Simple RNN            | 0.6325        |
| CNN + LSTM                  | 0.6581        |
| **CNN + Bi-LSTM** | **0.6838** |
| CNN + xLSTM (Conceptual)    | 0.6325        |
| CNN + GRU                   | 0.6581        |
| **CNN + LSTM + Attention** | **0.6752** |

The Bi-LSTM model achieved the highest overall accuracy, closely followed by the LSTM with Attention model.

### Detailed Performance Analysis

While overall accuracy provides a general picture, detailed classification reports and confusion matrices reveal significant insights into per-class performance, especially for minority classes (indexed 0-4 for `artifact`, `extrahls`, `extrastole`, `murmur`, `normal` respectively).

**Key Observations from Detailed Reports (e.g., from notebook output):**

* **Simple RNN**: Achieved high recall for the majority 'normal' class (Class 4, 99% recall) but completely failed on minority classes (e.g., Class 1 & 2 showing 0% precision/recall), misclassifying them as 'normal'.
* **LSTM/GRU**: Showed slight improvements in overall accuracy and some minor gains for certain minority classes (e.g., Class 3 recall increased), but still struggled significantly with 'extrahls' (Class 1) and 'extrasystole' (Class 2).
* **Bi-LSTM**: Demonstrated a notable improvement in classifying 'artifact' (Class 0, 88% recall, 0.82 F1-score), showcasing the benefit of bidirectional context. It was the best performer overall.
* **LSTM + Attention**: Also significantly improved 'artifact' classification (88% recall, 0.82 F1-score) and achieved very competitive overall accuracy, highlighting the mechanism's ability to focus on key audio segments.

## 8. Discussion

The experimental results offer crucial insights into the effectiveness of different deep learning architectures for heart sound classification:

* **Gated RNNs (LSTM, GRU) vs. Simple RNN**: The superior performance of LSTM and GRU over Simple RNN unequivocally demonstrates the necessity of gated recurrent units. These gates effectively mitigate vanishing gradients, enabling models to learn and retain long-term dependencies critical for understanding the sequential nature of heart sounds.
* **Bi-LSTM's Contextual Power**: The Bi-LSTM model consistently ranked highest in overall accuracy. This signifies that understanding the context from both preceding and succeeding audio segments is highly beneficial for accurate heart sound diagnosis. For example, the presence or absence of a sound before or after a suspected anomaly can be crucial for its correct classification.
* **Attention Mechanism's Role**: Integrating attention with LSTM provided a substantial performance boost, bringing its accuracy close to Bi-LSTM. Attention's ability to dynamically pinpoint and weigh the most informative parts of a long audio sequence is a powerful asset. This not only improves predictive accuracy but also offers a pathway to model interpretability, potentially highlighting to clinicians *which* specific parts of the heartbeat sound contributed most to the model's decision.
* **Persistent Class Imbalance**: Despite using stratified data splitting, a major challenge across all models was the severe class imbalance. Models consistently biased towards the majority 'normal' class, leading to very low (often zero) precision and recall for rare pathological classes like 'extrahls' and 'extrasystole'. This indicates that the models either didn't see enough examples of these rare conditions or couldn't sufficiently differentiate them from 'normal' or 'murmur' sounds with the current training strategy. For clinical deployment, this is a critical hurdle, as missing rare but significant conditions is unacceptable.

In summary, while advanced CNN-RNN architectures, particularly bidirectional and attention-enhanced models, show promise in learning complex audio patterns, the class imbalance remains a significant barrier to achieving robust and reliable performance across all heart sound categories.

## 9. Future Work

To address the identified limitations and enhance the model's clinical utility, the following future directions are proposed:

* **Addressing Class Imbalance**:
    * **Data Augmentation**: Employ advanced audio data augmentation techniques like pitch shifting, time stretching, adding realistic synthetic noise, or even generating synthetic minority class samples using Generative Adversarial Networks (GANs) or Variational Autoencoders (VAEs).
    * **Resampling Strategies**: Implement intelligent oversampling (e.g., SMOTE for features) or undersampling strategies to balance the dataset effectively.
    * **Weighted Loss Functions**: Introduce class weighting in the `CrossEntropyLoss` to heavily penalize misclassifications of minority classes, forcing the model to pay more attention to them.
* **Advanced Model Architectures**:
    * **Deeper CNN/RNN Models**: Explore deeper or more complex CNN backbones and RNN layers, potentially incorporating residual or dense connections to facilitate gradient flow.
    * **Transformer Networks**: Investigate transformer-based architectures for sequence modeling, as they have shown excellent performance in handling long-range dependencies and complex attention mechanisms in other domains.
    * **Hierarchical Classification**: Design a two-stage classification system (e.g., first classify as 'normal' or 'abnormal', then further classify 'abnormal' into specific pathologies).
* **Feature Engineering Refinement**:
    * **Diverse Audio Features**: Experiment with other audio features such as Chromagram, Spectral Contrast, or higher-order MFCCs to capture different acoustic properties.
    * **Learnable Feature Extractors**: Utilize end-to-end learnable feature extractors (e.g., SincNet) that are optimized during training, potentially outperforming fixed MFCCs.
* **Hyperparameter Optimization**: Conduct more extensive hyperparameter tuning (e.g., hidden sizes, learning rates, dropout rates, batch sizes) using automated techniques like Grid Search, Random Search, or Bayesian Optimization.
* **Ensemble Methods**: Combine the predictions from multiple top-performing models (e.g., Bi-LSTM and Attention-LSTM) using ensembling techniques (bagging, boosting, stacking) to improve overall robustness and accuracy.
* **Interpretability**: Continue exploring and visualizing attention weights to provide more actionable insights for medical professionals, fostering trust and clinical adoption.
* **Clinical Validation**: Once a sufficiently robust model is developed, rigorous validation with an independent, clinically diverse dataset is essential to assess real-world performance and generalizability.

## References

* **Project Notebook**:
    * [Mahla Entezari. (2025). ADM_3rdAssignment_MahlaEntezari.ipynb. *Kaggle Notebook*.](https://www.kaggle.com/mahlaentezari/adm-3rdassignment-mahlaentezari)
* **Dataset Sources**:
    * [Abdallah Aboelkhair. (2023). Heartbeat Sound. *Kaggle*.](https://www.kaggle.com/datasets/abdallahaboelkhair/heartbeat-sound)
    * [Kinguistics. (2023). Heartbeat Sounds. *Kaggle*.](https://www.kaggle.com/datasets/kinguistics/heartbeat-sounds)
    * [Mahla Entezari. (2024). set-a-b. *Kaggle*.](https://www.kaggle.com/datasets/mahlaentezari/set-a-b)
