# MediaPipe Deepfake Detection Prototype

Prototype for comparing image-level deepfake detection features across identity,
geometry, and frequency signals.

The project is organized around an ablation study: train the same classifier on
different feature combinations and compare performance on the same dataset split.

## Project Goal

Compare deepfake detection performance across generator families:

- GAN-based / face-swap style generators, for example SimSwap
- Autoencoder-based generators, for example DeepFaceLab
- Diffusion-based generators, for example Stable Diffusion / LDM

The feature groups are:

- **Identity features**: ArcFace embeddings from InsightFace
- **Geometry features**: MediaPipe facial landmarks summarized by facial region
- **Frequency features**: DCT energy statistics from the image

The classifier used by the experiment scaffold is XGBoost.

## Main Workflow

1. Gather real and fake images.
2. Create a dataset manifest CSV with image paths and labels.
3. Run `main.py` to execute experiments 1-7 in order. This trains and evaluates
   each classifier on the dataset split in the manifest.
4. Compare the printed accuracy tables and the generated `summary_*.csv`,
   `metrics.json`, `train_predictions.csv`, and `test_predictions.csv` files.

Example manifest:

```csv
image_path,label,split,generator,generator_type
real/img001.jpg,0,train,real,real
fake/simswap001.jpg,1,train,SimSwap,gan-based
fake/deepfacelab001.jpg,1,test,DeepFaceLab,autoencoder-based
fake/stablediffusion001.jpg,1,test,Stable Diffusion,diffusion-based
```

`label` uses `0 = real` and `1 = fake`.

## Important Commands

Install dependencies:

```bash
pip install -r requirements.txt
```

On macOS, XGBoost also needs the OpenMP runtime:

```bash
brew install libomp
```

Run all ablation experiments in order:

```bash
python3 main.py --manifest data/manifest.csv
```

Run one specific experiment from the main entry point:

```bash
python3 main.py --experiment 01_identity --manifest data/manifest.csv
python3 main.py --experiment 02_frequency --manifest data/manifest.csv
python3 main.py --experiment 03_geometry --manifest data/manifest.csv
python3 main.py --experiment 07_all_features --manifest data/manifest.csv
```

Experiment outputs are saved under the selected output directory in one folder
per experiment, for example `outputs/experiments/02_frequency/`. Use
`--output-dir` to choose a different parent folder:

```bash
python3 main.py \
  --experiment 02_frequency \
  --manifest data/manifest.csv \
  --output-dir outputs/frequency_debug
```

Experiments 3, 5, 6, and 7 use MediaPipe geometry features. If MediaPipe is not
installed in your environment, run the non-geometry experiments individually:

```bash
python3 -m experiments.experiment_01_identity --manifest data/manifest.csv
python3 -m experiments.experiment_02_frequency --manifest data/manifest.csv
python3 -m experiments.experiment_04_identity_frequency --manifest data/manifest.csv
```

Run just the identity-only experiment:

```bash
python3 -m experiments.experiment_01_identity --manifest data/manifest.csv
```

Run the all-features experiment:

```bash
python3 -m experiments.experiment_07_all_features --manifest data/manifest.csv
```

Run a trained model on one image:

```bash
python3 -m model_tools.image_inference path/to/image.jpg --model outputs/experiments/07_all_features/model.joblib
```

Optional: evaluate an already-trained model on a separate external test manifest.
This is not required for the main experiment workflow:

```bash
python3 -m model_tools.evaluate_model \
  --model outputs/experiments/07_all_features/model.joblib \
  --manifest data/test_manifest.csv \
  --output outputs/experiments/07_all_features/external_test_predictions.csv
```

Run the current unit tests:

```bash
python3 -m unittest tests.test_frequency tests.test_arcface_embedding
```

## Repository Files

### Root

- `README.md`
  - This project overview and file guide.

- `PROJECT_CONTEXT.md`
  - Research notes describing the larger project goal, feature groups, classifier
    plan, dataset assumptions, and ablation strategy.

- `requirements.txt`
  - Python dependencies for feature extraction, model training, evaluation, and
    image processing.

- `main.py`
  - Main experiment entry point.
  - Runs all seven ablation experiments in order.
  - Prints overall test accuracy, test accuracy by generator, and test accuracy
    by generator type.
  - Saves `summary_overall.csv`, `summary_by_generator.csv`, and
    `summary_by_generator_type.csv` in the experiment output directory.

### Feature Extractors

- `feature_extractors/frequency.py`
  - Extracts frequency-domain features from a single image.
  - Uses grayscale conversion, resizing, 2D DCT, low/mid/high frequency energy,
    global DCT standard deviation, and high/low energy ratio.
  - Can be imported as `extract_frequency_features(image)` or run as a CLI.

- `feature_extractors/arcface_embedding.py`
  - Extracts identity features from a single image using InsightFace ArcFace.
  - Detects the largest face, uses the `buffalo_l` model pack, and returns a
    512-dimensional normalized embedding as `arcface_000` through `arcface_511`.
  - Also supports optional face metadata such as bounding box, keypoints, and
    detection score.
  - Caches downloaded InsightFace models in the local `.insightface/` folder.

- `feature_extractors/feature_extraction.py`
  - Shared feature-combining layer.
  - Knows the three feature groups: `identity`, `frequency`, and `geometry`.
  - Creates reusable model context, such as one ArcFace analyzer per run.
  - Combines selected feature extractors into one dictionary for an image.
  - Calls `feature_extractors/mediapipe_features.py` when geometry experiments
    are used.

- `feature_extractors/mediapipe_features.py`
  - Extracts MediaPipe face landmarks, normalizes them by nose position and eye
    distance, splits them into facial regions, and summarizes those regions into
    numeric geometry features for the experiment pipeline.
  - Keeps manual debugging helpers such as landmark drawing and region
    comparison separate from the non-visual training path.

### Training

- `training/experiment_runner.py`
  - Shared training engine for all ablation experiments.
  - Loads a manifest CSV, extracts the requested feature groups, builds a feature
    table, trains a fresh XGBoost classifier, evaluates it, and saves outputs.
  - Each experiment trains its own classifier from scratch.
  - Outputs include `features.csv`, `model.joblib`, `metrics.json`,
    `train_predictions.csv`, and `test_predictions.csv`.

### Model Tools

- `model_tools/evaluate_model.py`
  - Optional extra batch evaluation script for an already-trained model.
  - Use this when you have a saved `model.joblib` and want to run it on a new test
    manifest without retraining.
  - Writes a prediction CSV and a companion metrics JSON file.
  - This is not part of the required 1-7 experiment workflow. The experiment files
    already train and evaluate their classifiers on the manifest's train/test split.

- `model_tools/image_inference.py`
  - Single-image inference entry point.
  - Loads a trained experiment model, extracts the same feature groups that model
    was trained on, aligns the feature columns, and prints the fake probability.

### Tests

- `tests/test_frequency.py`
  - Unit tests for the frequency feature extractor.
  - Useful for checking that `feature_extractors/frequency.py` behaves
    consistently.

- `tests/test_arcface_embedding.py`
  - Unit tests for the ArcFace identity extractor.
  - Checks embedding shape, finite values, unit norm, face bounding box,
    keypoints, and detection score on the sample images.

### MediaPipe Prototype Files

These files support MediaPipe landmark prototyping and the geometry extractor.

- `prototypes/mediapipe_landmarks/FacialLandmarkPrototype.py`
  - Current prototype for experimenting with MediaPipe face landmark detection
    and visualization.
  - This is not imported by the experiment pipeline.

- `prototypes/mediapipe_landmarks/facial_regions.py`
  - Region index lists for MediaPipe face landmarks, such as eyes, lips, nose,
    jawline, and cheeks.
  - Used by `feature_extractors/mediapipe_features.py`.

- `prototypes/mediapipe_landmarks/face_landmarker.task`
  - MediaPipe task model file used by the landmark prototype.

- `prototypes/mediapipe_landmarks/FacialLandmarkData/`
  - Local sample data for landmark prototyping.

### Experiment Entry Files

The files in `experiments/` are intentionally small wrappers around
`training/experiment_runner.py`. Their purpose is to give each ablation
experiment a clear command and output folder while keeping the actual training
logic in one shared place.

- `experiments/__init__.py`
  - Marks `experiments/` as a Python package so experiments can be run with
    `python3 -m experiments.<experiment_file>`.

- `experiments/experiment_01_identity.py`
  - Runs experiment 1: identity features only.
  - Uses ArcFace embeddings.

- `experiments/experiment_02_frequency.py`
  - Runs experiment 2: frequency features only.
  - Uses DCT features from `feature_extractors/frequency.py`.

- `experiments/experiment_03_geometry.py`
  - Runs experiment 3: geometry features only.
  - Uses MediaPipe features from `feature_extractors/mediapipe_features.py`.

- `experiments/experiment_04_identity_frequency.py`
  - Runs experiment 4: identity + frequency features.

- `experiments/experiment_05_identity_geometry.py`
  - Runs experiment 5: identity + geometry features.
  - Uses MediaPipe features from `feature_extractors/mediapipe_features.py`.

- `experiments/experiment_06_frequency_geometry.py`
  - Runs experiment 6: frequency + geometry features.
  - Uses MediaPipe features from `feature_extractors/mediapipe_features.py`.

- `experiments/experiment_07_all_features.py`
  - Runs experiment 7: identity + frequency + geometry features.
  - Uses MediaPipe features from `feature_extractors/mediapipe_features.py`.

### Data Files

- `data/test_images/sample1.jpg`
  - Sample image for manually testing feature extractors.

- `data/test_images/sample2.jpg`
  - Sample image for manually testing feature extractors.

- `data/test_images/sample3.jpg`
  - Sample image for manually testing feature extractors.

- `data/test_images/sample4.jpg`
  - Sample image for manually testing feature extractors.

- `data/test_images/sample5.jpg`
  - Sample image for manually testing feature extractors.

Future dataset files should live under `data/` or another documented dataset
folder. The experiment runner expects a manifest CSV rather than scanning folders
automatically.

### References

The `references/facial_embedding/deepfakes/` folder contains the earlier
video-based ArcFace experiment that inspired the image-only
`feature_extractors/arcface_embedding.py` module.

- `references/facial_embedding/deepfakes/README.md`
  - Documentation for the original latent-space outlier detection notebook.

- `references/facial_embedding/deepfakes/requirements.txt`
  - Dependencies for the original notebook experiment.

- `references/facial_embedding/deepfakes/deepfake_detection.ipynb`
  - Source notebook for the video-based ArcFace deepfake experiment.
  - Processes videos, extracts frames, computes ArcFace embeddings, visualizes
    them, and detects swapped-frame outliers.

- `references/facial_embedding/deepfakes/deepfake_detection_executed.ipynb`
  - Executed version of the same notebook with saved outputs.

- `references/facial_embedding/deepfakes/outputs/manipulated.mp4`
  - Generated sample manipulated video from the notebook experiment.

- `references/facial_embedding/deepfakes/outputs/manipulated_annotated.mp4`
  - Annotated version of the manipulated video with face boxes, keypoints, and
    detection labels.

- `references/facial_embedding/deepfakes/outputs/umap.png`
  - UMAP visualization of ArcFace embeddings from the notebook.

- `references/facial_embedding/deepfakes/outputs/histogram.png`
  - Histogram of identity-distance scores from the notebook.

## Generated Outputs

Running experiments creates folders like:

```txt
outputs/experiments/01_identity/
outputs/experiments/02_frequency/
outputs/experiments/07_all_features/
```

Each experiment folder contains:

- `features.csv`
  - Extracted feature table for that experiment.

- `model.joblib`
  - Trained classifier for that exact feature combination.

- `metrics.json`
  - Training and testing metrics, including accuracy, precision, recall, F1,
    ROC-AUC, average precision, and grouped summaries when generator metadata is
    available.

- `train_predictions.csv`
  - Per-image training predictions and fake probabilities.

- `test_predictions.csv`
  - Per-image test predictions and fake probabilities.

These files are produced by the experiment commands themselves, so the main
project comparison does not require `model_tools/evaluate_model.py`.

## Notes

- Each experiment retrains a separate classifier from scratch.
- Saved models are only reused for later inference or external testing.
- Avoid naming the geometry file `mediapipe.py`; use
  `feature_extractors/mediapipe_features.py` so it does not shadow the installed
  `mediapipe` package.
- `.insightface/` is a local generated model cache and is ignored by git.
