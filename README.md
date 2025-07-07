# Metacognition Prediction in Human-AI Collaboration

## Overview

This repository contains the code and data for the manuscript "Metacognition Prediction in Human-AI Collaboration" submitted to IEEE Transactions on Artificial Intelligence (IEEE TAI). This research investigates the prediction of human metacognitive processes during collaborative tasks with AI systems, combining behavioral analysis and electroencephalography (EEG) data.

## Abstract

Human-AI collaboration has become increasingly prevalent across various domains, yet understanding and predicting human metacognitive processes during these interactions remains a significant challenge. This study presents a novel approach to predict metacognitive states in human-AI collaborative environments using multimodal data analysis, including behavioral metrics and neural signals. Our methodology combines machine learning techniques with EEG signal processing to provide real-time insights into human metacognitive processes, enabling more effective and adaptive AI systems.

## Research Objectives

- **Metacognition Prediction**: Develop predictive models for human metacognitive states during AI collaboration
- **Multimodal Analysis**: Integrate behavioral and neural data for comprehensive metacognition assessment
- **Real-time Processing**: Enable real-time prediction of metacognitive states for adaptive AI systems
- **Collaboration Enhancement**: Improve human-AI collaboration through better understanding of metacognitive processes

## Repository Structure

```
ieee_tai/
├── code/
│   ├── preprocessing_epoching_data_for_analysis/    # Data preprocessing and epoching scripts
│   └── machine_learning_model_experiments/          # ML model development and experiments
├── data/
│   ├── behaviour/                                   # Behavioral data and metrics
│   └── epoched_eeg_and_behaviour_features/         # Processed EEG and behavioral features
└── README.md                                        # This file
```

## Data Description

### Behavioral Data (`data/behaviour/`)
Contains behavioral metrics collected during human-AI collaborative tasks, including:
- Task performance metrics
- Response times and accuracy
- Confidence ratings
- Collaboration patterns

### EEG Data (`data/epoched_eeg_and_behaviour_features/`)
Processed electroencephalography data and extracted features:
- Epoched EEG signals
- Spectral features
- Connectivity measures
- Behavioral feature integration

## Code Organization

### Preprocessing (`code/preprocessing_epoching_data_for_analysis/`)
Scripts for data preprocessing and preparation:
- EEG signal preprocessing
- Behavioral data cleaning
- Feature extraction

### Machine Learning Experiments (`code/machine_learning_model_experiments/`)
Implementation of predictive models:
- Training procedures
- Evaluation metrics
- Cross-validation protocols

## Key Contributions

1. **Novel Metacognition Prediction Framework**: Development of a comprehensive framework for predicting metacognitive states in human-AI collaboration
2. **Multimodal Data Integration**: Integration of behavioral and neural data for enhanced prediction accuracy
3. **Real-time Processing Pipeline**: Implementation of real-time data processing and prediction capabilities
4. **Adaptive AI System Design**: Guidelines for designing AI systems that adapt to human metacognitive states

