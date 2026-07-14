# Research Decisions and Methodology Notes

This document records the reasoning, formulas, and planned analyses discussed
during development. It is intended as a source for the research paper. Items
are marked as **implemented**, **decided**, **planned**, or **open** so that a
planned method is not accidentally reported as completed work.

## 1. Research objective

**Decided.** Compare image-level deepfake detection signals across three feature
families and across GAN/face-swap, autoencoder, and diffusion generator types:

- Identity: ArcFace embeddings.
- Geometry: normalized MediaPipe facial landmarks summarized by region.
- Frequency: five DCT statistics (low-, mid-, and high-frequency energy, global
  standard deviation, and the high/low-energy ratio).

This is a comparative experimental prototype, not a production detector. All
methods must be evaluated on the same split so differences can be attributed to
the feature/model choice rather than different samples.

## 2. Baseline feature experiments

**Implemented.** Experiments 1-7 train separate XGBoost classifiers using:

1. Identity only.
2. Frequency only.
3. Geometry only.
4. Identity + frequency.
5. Identity + geometry.
6. Frequency + geometry.
7. Identity + frequency + geometry.

Combined experiments concatenate feature columns before fitting one XGBoost
model. This is feature-level (early) fusion. It is distinct from combining the
probabilities of independently trained classifiers (decision-level/late
fusion).

XGBoost is based on threshold splits within individual columns. Consequently,
raw differences in numerical scale do not inherently give a DCT column more
influence than an ArcFace column. Standardization is therefore not required for
the current XGBoost baseline, although it is required for the planned
distance-based centroid analysis.

## 3. Geometry feature behavior

**Implemented behavior.** MediaPipe returns one complete 478-landmark face mesh
when a face is detected. The extractor does not detect facial regions
independently. It indexes fixed landmark subsets and produces ten statistics per
configured region: centroid `(x, y, z)`, standard deviation `(x, y, z)`, span
`(x, y, z)`, and centroid distance. It also produces whole-face statistics.

The extraction outcome is therefore:

- Face detected: all 478 landmarks and all configured regions are used.
- Face not detected: geometry values are missing (`NaN`).

`geometry_landmark_count` is consequently 478 for every successful detection
and has no discriminative variance. **Planned:** remove it or replace it with an
explicit `geometry_face_detected` indicator, and report face-detection failure
rates separately for real and fake images. This check is necessary because a
class-dependent detection-failure rate could become an unintended shortcut.

The current region definitions overlap (for example, `eyes` overlaps the two
individual eye regions and `face_oval` overlaps parts of the jawline). A region
ablation should therefore use interpretable conceptual groups rather than claim
that every existing entry is an independent anatomical signal.

## 4. Evaluation definitions and priorities

**Decided.** Fake is the positive class (`label = 1`) and real is the negative
class (`label = 0`). Missing a fake is the primary failure:

- True positive (TP): fake predicted as fake.
- False negative (FN): fake predicted as real; primary failure.
- False positive (FP): real predicted as fake.
- True negative (TN): real predicted as real.

The confusion-matrix layout used by scikit-learn is:

```text
[[TN, FP],
 [FN, TP]]
```

The implemented metrics are:

\[
\text{Accuracy}=\frac{TP+TN}{TP+TN+FP+FN}
\]

\[
\text{Precision}=\frac{TP}{TP+FP}
\]

\[
\text{Recall}=\frac{TP}{TP+FN}
\]

\[
\text{False-negative rate}=\frac{FN}{TP+FN}=1-\text{Recall}
\]

\[
F_1=2\frac{\text{Precision}\cdot\text{Recall}}
{\text{Precision}+\text{Recall}}
\]

Accuracy, precision, recall, F1, ROC AUC, average precision, and the confusion
matrix are already saved for the training and test sets. Test metrics are the
experimental results; training metrics are diagnostic evidence for
underfitting/overfitting because they are calculated after fitting on the same
training samples.

Recall and false-negative rate are the primary operational measures because an
undetected fake is the stated main failure. Recall cannot be optimized without
a constraint: predicting every image as fake gives perfect recall but unusable
precision and false-positive rate. The intended decision rule is therefore to
maximize validation recall subject to a predefined acceptable false-positive
rate (or minimum precision).

ROC AUC measures ranking quality across classification thresholds. It can be
interpreted as the probability that a randomly chosen fake receives a higher
fake score than a randomly chosen real. Average precision summarizes the
precision-recall trade-off across thresholds. These are useful
threshold-independent comparisons between feature experiments, but they do not
replace reporting recall at the selected operating threshold.

**Planned:** add false-negative rate and recall-weighted F2:

\[
F_2=5\frac{\text{Precision}\cdot\text{Recall}}
{4\cdot\text{Precision}+\text{Recall}}
\]

Also add ROC, precision-recall, and confusion-matrix plots.

## 5. Data splitting and threshold selection

**Decided methodology.** Model fitting, configuration selection, and final
evaluation must use separate data roles:

- Training: fit the three base classifiers and any preprocessing parameters.
- Validation: select fusion weights and the final classification threshold.
- Test: perform one final unbiased evaluation after choices are fixed.

The test set must not be used to choose weights, thresholds, feature subsets, or
other hyperparameters. The acceptable false-positive-rate ceiling (or minimum
precision) must be chosen before examining final test results. Its numerical
value remains **open** and should be justified by the intended use case rather
than selected for the best test score.

Generated outputs should preserve a stable `sample_id` and explicit split. The
current `train_predictions.csv` and `test_predictions.csv` contain predictions
for the exact already-assigned sets; they do not predict split membership.
Stable IDs will make joins unambiguous if an image path occurs more than once.

## 6. Centroid and cosine-similarity analysis

**Planned.** This is a post-experiment explanatory analysis, not the XGBoost
training procedure. Feature vectors are read from `features.csv`; the exact
training and test memberships are recovered from saved split information.

Preprocessing must occur after the split and before centroid calculation:

1. Fit median imputation on training columns only.
2. Fit column standardization on the imputed training columns only.
3. Apply those fitted transformations to both training and test vectors.
4. Calculate real and fake centroids using standardized training vectors only.
5. Compare each standardized test vector with both centroids.

For training-column mean \(\mu_j\) and standard deviation \(\sigma_j\):

\[
z_{ij}=\frac{x_{ij}-\mu_j^{train}}{\sigma_j^{train}}
\]

For standardized training vectors, the class centroids are:

\[
c_{real}=\frac{1}{N_{real}}\sum_{i:y_i=0}z_i,
\qquad
c_{fake}=\frac{1}{N_{fake}}\sum_{i:y_i=1}z_i
\]

Cosine similarity is:

\[
\cos(z,c)=\frac{z\cdot c}{\lVert z\rVert\lVert c\rVert}
\]

Each test image receives similarity-to-real and similarity-to-fake values. A
signed explanatory score may be defined as:

\[
s(z)=\cos(z,c_{fake})-\cos(z,c_{real})
\]

Positive values indicate greater similarity to the fake centroid and negative
values greater similarity to the real centroid. Explicit per-vector L2
normalization is optional because cosine similarity already divides by vector
magnitude; column standardization is the essential step.

For combined experiments, standardizing columns prevents domination caused by
raw numerical scale but not domination caused by dimensionality (for example,
hundreds of ArcFace columns versus five frequency columns). **Decided:** compute
centroids and similarity scores separately for identity, frequency, and geometry
and combine available group scores with equal weights:

\[
s_{combined}=\frac{s_{identity}+s_{frequency}+s_{geometry}}{3}
\]

Report the group-specific distributions as the main explanatory result. A raw
concatenated-space cosine may be shown secondarily, but it must not be described
as giving feature families equal influence.

## 7. Frequency features and JPEG compression

**Observed limitation.** Heavy JPEG compression can destroy high-frequency
information and make the DCT feature vector less useful. A JPEG quality label or
estimated quality can describe/stratify degradation, but it cannot recover
discarded information. Re-saving a degraded image at high quality also cannot
restore it.

**Decided dataset design.** The dataset will contain equal numbers of two
source-defined quality groups:

- `high`: PNG images gathered from datasets that provide high-quality PNGs.
- `low`: visibly compressed JPEG images gathered from datasets that provide
  lower-quality images.

Images are stored in separate `high` and `low` directories and the quality group
must also be recorded in the manifest. These are operational dataset labels,
not measured or recovered JPEG encoder quality factors. PNG format alone does
not prove that an image was never compressed before being saved, and an existing
JPEG generally does not reveal a unique original encoder quality setting.

This collected-group design measures whether DCT features perform differently
on the selected high- and low-quality data. It does **not** isolate JPEG
compression as the sole causal variable as a paired experiment would, because
quality group, file format, source dataset, image content, resolution, and
generator distribution may be correlated. Consequently, the supported paper
claim is that DCT performance decreased for the collected low-quality group,
not that a particular compression level alone caused the decrease.

To make the comparison defensible:

- Document the source dataset and inclusion criteria for every image.
- Balance high/low counts within real/fake classes and, where possible, within
  generator types rather than only balancing the dataset globally.
- Preserve both quality groups across training, validation, and test splits.
- Record objective metadata such as format, width, height, file size, and bytes
  per pixel to describe the groups alongside the visual/source-based label.
- Report results separately by quality group and source dataset when sample
  sizes permit.

Validation data used for fusion must represent both collected quality groups;
otherwise it may assign excessive weight to the frequency classifier.

A future paired recompression experiment using identical source images would be
a stronger causal test, but it is not part of the current dataset plan.

A single global fusion weight set is preferred for the initial study. A
quality-aware rule that changes DCT weight per image is deferred because it adds
another estimator and requires separate validation.

## 8. Feature ablations

**Planned supporting experiments.** Ablation complements rather than replaces
the seven baseline comparisons.

Frequency ablation can reuse an extracted feature table and select columns
without recomputing DCTs. There are 31 non-empty subsets of five features, but a
clear initial analysis is:

- All five features as baseline.
- Five leave-one-feature-out runs.
- Five single-feature runs.
- Exhaustive subsets only as optional exploratory analysis.

The DCT variables are not fully independent: the high/low ratio is derived from
high and low energy, and global DCT standard deviation relates to overall
energy. Results should therefore be interpreted as contribution analyses, not
as effects of independent variables.

Geometry ablation should remove all columns belonging to a conceptual region
group together. Recommended groups are eyes, eyebrows, nose, mouth, cheeks, and
face contour. Compare all groups, leave-one-group-out, each group alone, and
whole-face versus regional summaries. Avoid all subsets of the 17 overlapping
dictionary entries because this creates 131,071 combinations with unclear
interpretation.

Every ablation must reuse the same data split, classifier configuration, and
evaluation protocol.

## 9. Decision-level classifier fusion

**Decided direction.** In addition to experiment 7's concatenated-feature
classifier, combine the independently trained identity, frequency, and geometry
base classifiers at the probability level. “Base” or “specialist classifier” is
preferred over “weak classifier,” because the individual XGBoost models are not
weak learners in the formal boosting sense.

Equal-weight soft voting is the low-cost baseline:

\[
p_{equal}=\frac{p_{identity}+p_{frequency}+p_{geometry}}{3}
\]

The proposed method is validation-selected weighted soft voting:

\[
p_{fused}=w_Ip_I+w_Fp_F+w_Gp_G
\]

subject to:

\[
w_I,w_F,w_G\geq0,
\qquad
w_I+w_F+w_G=1
\]

Use a small predefined grid of weights, evaluate each combination on validation
data, and select the weights and classification threshold that maximize fake
recall while satisfying the predefined false-positive constraint. This allows
representative compressed validation data—not intuition or test results—to
determine whether the frequency classifier deserves less influence.

Experiments 1-3 can be reused for equal/weighted voting if they use identical
sample splits. Join their validation/test fake probabilities by stable sample ID
and verify that IDs and true labels match before fusion.

Stacking is intentionally deferred. It would require leakage-safe out-of-fold
base-model predictions and a trained meta-classifier, adding complexity beyond
the present research need.

The final comparison should include:

- Each individual base classifier.
- Experiment 7 feature-level concatenation.
- Equal-weight soft voting.
- Validation-selected weighted soft voting.

The fused model should only be called “stronger” if it improves the chosen test
criteria, especially recall/false-negative rate under the false-positive
constraint.

## 10. Remaining decisions and implementation checklist

- Define and justify the acceptable false-positive-rate ceiling or minimum
  precision before final testing.
- Add a validation split while preserving class and generator representation.
- Preserve `sample_id` and split assignments in all relevant outputs.
- Add false-negative rate, F2, ROC, precision-recall, and confusion-matrix
  reporting/plots.
- Remove constant `geometry_landmark_count` or replace it with face-detection
  status; report detection failures by class.
- Implement training-only-standardized centroid/cosine analysis and histograms.
- Implement equal and validation-weighted soft-voting fusion.
- Evaluate performance separately for the collected high- and low-quality
  groups and document their selection criteria and source-dataset confounds.
- Run frequency and conceptual-region geometry ablations after the baseline
  experiments are established.
