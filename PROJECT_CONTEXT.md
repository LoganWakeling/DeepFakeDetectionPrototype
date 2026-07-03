#Deepfake Detection Research Project

## Goal
Compare deepfake detection performance across 3 generator types:
- Gan-based (e.g., SimSwap, DeepFaceLab)
- Autotencoder-based
- Diffusion-based (e.g., Stable Diffusion / LDM)

## Feature Groups (3 metrics)

### 1. Identity features
- ArcFace embeddings

### 2. Geometric features
- MediaPipe face landmarks
- distances + angles between facial points

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
- Frequency features are extracted from the full image (resized, not cropped), ensuring consistency with identity and geometry pipelines.