# Speech Emotion Recognition Datasets

This folder contains four widely used datasets for Speech Emotion Recognition (SER):

1. [RAVDESS](#ravdess)
2. [SAVEE](#savee)
3. [CREMA-D](#crema-d)
4. [TESS](#tess)

These datasets are valuable for training, validating, and benchmarking SER models.

---

## Table of Contents

- [Overview](#overview)  
- [RAVDESS](#ravdess)    
- [SAVEE](#savee)   
- [CREMA-D](#crema-d)    
- [TESS](#tess)  
- [How to Use](#how-to-use)  
- [References](#references)  
- [License](#license)

---

## Overview

Speech Emotion Recognition (SER) focuses on detecting emotions from audio recordings of human speech. Emotions such as **angry**, **happy**, **sad**, **neutral**, **fear**, **disgust**, and **surprise** are typically labeled in these datasets, though the exact emotions available may vary depending on the dataset.

In this folder, you will find four widely used datasets in the SER community:

1. **RAVDESS** (Ryerson Audio-Visual Database of Emotional Speech and Song)  
2. **SAVEE** (Surrey Audio-Visual Expressed Emotion)  
3. **CREMA-D** (Crowd-Sourced Emotional Multimodal Actors Dataset)  
4. **TESS** (Toronto Emotional Speech Set)

---

## RAVDESS

- **Full Name**: Ryerson Audio-Visual Database of Emotional Speech and Song  
- **Source**: [Official RAVDESS Website](https://zenodo.org/record/1188976)  
- **Details**:  
  - 24 professional actors (12 male, 12 female)  
  - 2 speech and 2 song modalities  
  - 8 different emotions (neutral, calm, happy, sad, angry, fearful, surprise, and disgust)  
  - Each expression is recorded at two levels of emotional intensity

---

## SAVEE

- **Full Name**: Surrey Audio-Visual Expressed Emotion  
- **Source**: [Official SAVEE Website](http://kahlan.eps.surrey.ac.uk/savee/)  
- **Details**:  
  - Recorded from 4 male actors  
  - 7 different emotions (anger, disgust, fear, happiness, neutral, sadness, surprise)  
  - Audio files named using a code that indicates actor, emotion, and index

---

## CREMA-D

- **Full Name**: Crowd-Sourced Emotional Multimodal Actors Dataset  
- **Source**: [Official CREMA-D GitHub](https://github.com/CheyneyComputerScience/CREMA-D)  
- **Details**:  
  - 91 actors (48 male, 43 female)  
  - 6 emotions (angry, disgust, fear, happy, neutral, sad)  
  - Crowd-sourced validation of emotion labels

---

## TESS

- **Full Name**: Toronto Emotional Speech Set  
- **Source**: [Official TESS Website](https://tspace.library.utoronto.ca/handle/1807/24487)  
- **Details**:  
  - 2 female actors (aged 26 and 64)  
  - 7 different emotions (anger, disgust, fear, happy, neutral, pleasant surprise, sad)  
  - Each actor produced the same set of target words

---

## How to Use

1. **Clone or download** the dataset.  
2. **Preprocessing**: Convert audio files to a consistent sample rate, normalize volume, or extract features (e.g., MFCCs, log Mel spectrograms) as required.  
3. **Model training**: Use the preprocessed audio data to train and evaluate your SER models.  
4. **Evaluation**: Compare model performance across different datasets to further generalize your model's performance.

---

## References

[1] S. R. Livingstone and F. A. Russo, “The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS),” PLOS ONE, vol. 13, no. 5, p. e0196391, May 2018. [Online]. Available: https://doi.org/10.1371/journal.pone.0196391

[2] S. Haq and P. J. B. Jackson, “Surrey Audio-Visual Expressed Emotion (SAVEE) Database," April 2 2015. [Online]. Available: http://kahlan.eps.surrey.ac.uk/savee/ 

[3] H. Cao, D. G. Cooper, M. K. Keutmann, R. C. Gur, A. Nenkova, and R. Verma, “CREMA-D: Crowd-Sourced Emotional Multimodal Actors Dataset,” IEEE Transactions on Affective Computing, 2014. [Online]. Available:
https://github.com/CheyneyComputerScience/CREMA-D

[4] J. A. Craig and S. A. Smith, “The Toronto Emotional Speech Set (TESS),” 2009. [Online]. Available:
https://tspace.library.utoronto.ca/handle/1807/24487

---

## License

- This repository is provided for **educational and research purposes**.  
- Each dataset may have its own licensing terms. **You must read and follow the license agreements** for each dataset before using or distributing it.  
- If you use any of these datasets, **please properly cite** the original creators and sources.

---

**Disclaimer**: While this repository provides an overview of these popular speech emotion datasets, always consult the official websites and publications for the most updated and accurate information. Make sure to follow the original creators’ guidelines on dataset usage and redistribution. 

---