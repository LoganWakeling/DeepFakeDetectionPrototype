# Deepfake Detection Research Project

## Goal
Compare deepfake detection performance across 3 generator types:
- Gan-based (e.g., SimSwap, DeepFaceLab)
- Autoencoder-based
- Diffusion-based (e.g., Stable Diffusion / LDM)

## Feature Groups (3 metrics)

### 1. Identity features
- ArcFace embeddings

### 2. Geometric features
- MediaPipe face landmarks
- normalized facial-region landmark statistics

### 3. Frequency features
- 2D DCT
- low/mid/high frequency energy
- global std + high/low ratio

## Model
- XGBoost classifier
- multiple ablation experiments:
    identity-only, geometry-only, frequency-only, and combinations

## Dataset
- Balanced across architectures
- Train/test split fixed across all experiments

## Notes
- This is an experimental comparison study, not a production detector
- A shared largest-face crop (25% margin) is applied before identity, geometry,
  and frequency extraction. If no face is detected, extraction falls back to
  the original image.
- Stable feature extraction code lives in `feature_extractors/`; the MediaPipe geometry extractor is `feature_extractors/mediapipe_features.py`.
