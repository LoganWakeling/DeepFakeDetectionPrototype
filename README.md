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

### Creating a manifest from a dataset folder

The dataset itself can remain outside this repository. From the repository
root, adapt `data/manifest_config.json` so its folder-name mappings match the
dataset, then run:

```bash
.venv/bin/python scripts/create_manifest.py "<dataset-root>" \
  --output data/manifest.csv \
  --config data/manifest_config.json \
  --strict
```

Replace `<dataset-root>` with the path to the database folder. The script
recursively scans it, records absolute image paths, reads objective image
properties, and infers labels from directory names. Use
`data/manifest_config.example.json` as a reference when adapting the active
configuration.

Matching is case-insensitive and works at any directory level. Built-in names
cover common `real`, `fake`, `high`, `low`, train/test, SimSwap, DeepFaceLab,
and Stable Diffusion folders. `--strict` prevents writing a manifest when a
real/fake label, quality label, or fake generator family cannot be inferred;
omit it for an initial scan that writes blanks and prints warnings. The dataset
root folder name is used as the source dataset unless a more specific mapped
directory is found. If train/test directories
are absent, a stable stratified 80/20 split is assigned (change it with
`--test-fraction`). Stratification jointly preserves class, generator type,
quality label, and source dataset proportions. Every stratum with at least two
images receives both train and test data; singleton strata remain in training
and are listed in the summary. Existing train/test directory assignments are
preserved. Re-running with the same relative file layout preserves sample IDs
and assignments.

Generated columns are `sample_id`, `image_path`, `label`, `class_name`,
`split`, `source_dataset`, `quality_label`, `generator`, `generator_type`,
`manipulation_type`, `image_format`, `width`, `height`, `channels`,
`file_size_bytes`, and `bytes_per_pixel`. Quality is the dataset's
source-defined label; the numeric properties are recorded separately and do
not claim to recover an image's prior compression history.
All manifest columns are preserved in the per-image prediction CSVs. Metrics
also include grouped summaries by generator, generator type, quality label, and
source dataset when those columns are populated.

The command also writes `data/manifest.summary.json` by default. It records
total image and byte counts, average dimensions, distributions for all major
metadata fields, the same distributions within each split, missing metadata
warnings, and singleton strata. Use `--summary another/path.json` to change its
location.

Stratification alone cannot prevent leakage when multiple images come from the
same person, source video, or real/fake pair. If the database has those
relationships, they should be exposed as a group identifier before splitting
so all related images stay in one split.

## Evaluation Goals and Metrics

Fake images (`label = 1`) are the positive class. The primary failure is a
**false negative**: a fake image classified as real. Therefore, fake-class
recall and the false-negative rate are the main operational measures. High
recall must be considered together with precision or the false-positive rate so
that it is not achieved simply by classifying nearly every image as fake.

The experiments currently save the following train and test metrics in
`metrics.json`:

- **Accuracy**: proportion of all images classified correctly.
- **Precision**: proportion of images predicted as fake that are actually fake.
- **Recall**: proportion of all fake images detected; `1 - recall` is the
  false-negative rate.
- **F1**: balance between precision and recall at the selected decision
  threshold.
- **ROC AUC**: ability to rank fake images above real images across decision
  thresholds.
- **Average precision**: summary of the precision-recall trade-off across
  thresholds.
- **Confusion matrix**: counts in scikit-learn order `[[TN, FP], [FN, TP]]`.

Test metrics are the main experimental results. Training metrics are reported
to help diagnose overfitting. ROC AUC and average precision compare overall
separability, while recall and false negatives reflect the project's primary
failure. A future improvement is to report F2 and select a decision threshold
on validation data that prioritizes recall while limiting false positives.

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

- `RESEARCH_DECISIONS.md`
  - Paper-oriented record of methodology decisions, formulas, rationale,
    planned analyses, and remaining open choices.

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

## Dataset

- Use equal numbers of source-defined `high`-quality PNG images and
  `low`-quality compressed JPEG images, stored in corresponding directories.
- Record the quality group and source dataset in the manifest. The `high` and
  `low` labels describe how the groups were collected; they are not measured
  JPEG quality factors.
- Balance quality groups within real/fake classes and generator types where
  possible, and preserve both groups in train, validation, and test splits.
- Record image format, dimensions, file size, and bytes per pixel so the groups
  can be described objectively in addition to their source-based labels.
- Interpret a DCT performance difference as an association with the collected
  quality groups. Because format and source dataset also differ, this design
  does not prove that JPEG compression alone caused the difference.

## Notes
- Histograms
- Display results to visualize
- Compute cosine similarity and display histograms
- Ablation for the 5 features of frequency.py (for every experiment) 2^5
- Careful on how images are stored so the original PNG/JPEG quality-group signal
  is not accidentally removed.
- Low-quality JPEGs may have already lost frequency information, so evaluate
  frequency performance separately for the two quality groups.
- Even if we have the quality factor (for example 100%), if its compressed, then restored back to 100%, the information is still lost
- Do not convert the collected JPEGs to PNG and describe them as restored
  high-quality images; loss from earlier compression is irreversible.
- Include the source-defined `high`/`low` quality group in the manifest.
- Ablation study on the number of regions for landmark 
- Arcface ablation study on upper/lower face
- Distribution of feature vectors for arcface, mediapipe, and dct should be normalized the same way (So the concatenated feature vector doesn't have numbers 1000+ for dct and 0-1 for arcface (bad distribution)).
- Training the classifiers separately (weak classifiers) and then fusing them (instead of concatenating the feature vectors).
- Consider the decision metrics, voting, etc.
- Define metrics of our experiments (not just accuracy) (have plots) (for example, precision, recall, TP/FP)
  - Our project already does this. evaluate_model() calculates accuracy, precision, recall, F1 score, ROC AUC, average precision, and confusion matrix(scikit-learn layout, TN, FP, FN, TP), saved in metrics.json.

- Use our dataset for our experiments, and then afterwards we can use the evaulation_model.py for external datasets.
- Test generalization, and we can show which feature extractor has the best generalization (mediapipe most likely).
- Train on our dataset, and test on different datasets (How can we make sure the results (for example, poor generalization) aren't because of our training dataset?)
- Train the classifier multiple times on different datasets before evaluation on external datasets (Separate experiment). 


## Cosine Similarity Computation for Analysis
- After we run the experiments and get our outputs, we can write a script that computes the average feature vector for all real images, and the average feature vector for all fake images (from the training dataset). Then, it will compute the cosine similarity between a test image (n) (in the test dataset) and the known real/fake centroid (the feature vector averages), for every n test image. Then we can display that on a histogram.
- We can do it separately, as well, for each feature group. 1. identity 2. geometry 3. frequency 4. all features
- That way, let's say, experiment 2 (frequency-only) performs well. A histogram may show: real images have high similarity to the real frequency centroid. Fake images shift away from that centroid.
- But let's say experiment 3 (geometry-only) performs poorly. The histogram would show that real and fake geoemtry similarities heavily overlap (their cosine similarities to the centroid are the same across the board)
- This would explain why the classifier struggles with geometry-only detection (for example).
- It would answer the questions:
- Do real and fake images occupy measurably different regions of feature space?
- Which feature group separates them most clearly?
- Do GAN, diffusion, and autoencoder fakes differ from real images in different ways?

- Note: the calculated real and fake centroids will be split into feature specifics. For example, we'll have an identity-feature real centroid, geometry-feature real centroid, and frequency-feature real centroid
- If we choose to plot the generator type (gan, diffusion, and autoencoder), it can show which generation type does a better job at creating images most similar to real images (based on feature-specificity of course).

- Generate script to calculate the centroids and cosine similarities AFTER dataset is created and ran through the model. (We need to finalize what the columns in our outputs look like.)

- We'll add a separate script post-experiment to calculate and display the histograms from the output we receive.

Before calculating centroids, the post-experiment script should split the rows
into the exact training and test sets, fit missing-value imputation and column
standardization on training features only, and apply those fitted transforms to
both sets. Real and fake centroids are averages of the standardized **training**
vectors; standardized test vectors are then compared with both centroids. For
combined experiments, calculate similarity per feature group and combine the
group scores with equal weights so that a high-dimensional group such as
ArcFace does not dominate simply because it has more columns.

## Feature Scaling
- The XGBoost classifier uses decision trees to split individual columns in the thresholds, so a DCT column being numerically larger than an ArcFace column does not inherently give it more influence.
- The concatenation of feature vectors may not need normalization (to ensure a balanced distribution among the vectors) because of that.
- geometry_landmark_count in the mediapipe feature extractor will always be 478 (if an image is detected), so removing this from the feature vector might be helpful. (Unless, we have it so each region is detected independently, for example, if there is a face with only the eyes visible, only that region is detected and outputted. But since our dataset only includes pictures of entire faces, this would be unnecessary).

## Evaluation To-Dos

- Add F2 and false-negative-rate reporting.
- Choose the operating threshold using validation data rather than the test set;
  prioritize fake recall subject to an acceptable false-positive rate.
- Add ROC and precision-recall plots plus a clearly labelled confusion matrix.
- Implement the post-experiment centroid/cosine-similarity analysis using only
  training-fitted preprocessing.
- Preserve an explicit sample identifier or split assignment in generated
  outputs so post-experiment analyses can reproduce the exact split safely.
- Report MediaPipe face-detection failure rates separately for real and fake
  images, and remove the constant `geometry_landmark_count` feature or replace
  it with an explicit detection-status indicator.
