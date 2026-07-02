import cv2
import numpy as np
from frequency import extract_frequency_features


def extract_frequency_features_cv2dct(img, size=128):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    resized = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    normalized = resized / 255.0
    normalized = normalized - np.mean(normalized)

    dct_coefficients = cv2.dct(normalized)

    coefficients = np.abs(dct_coefficients)
    rows, cols = np.indices(coefficients.shape)
    normalized_frequency = (rows + cols) / (2 * (size - 1))

    low_mask = normalized_frequency < 0.15
    mid_mask = (normalized_frequency >= 0.15) & (normalized_frequency < 0.40)
    high_mask = normalized_frequency >= 0.40

    low_energy = float(np.mean(coefficients[low_mask] ** 2))
    mid_energy = float(np.mean(coefficients[mid_mask] ** 2))
    high_energy = float(np.mean(coefficients[high_mask] ** 2))

    return {
        "dct_low_energy": low_energy,
        "dct_mid_energy": mid_energy,
        "dct_high_energy": high_energy,
        "dct_global_std": float(np.std(dct_coefficients)),
        "dct_high_low_ratio": float(high_energy / (low_energy + 1e-8)),
    }


img = cv2.imread("data/test_images/sample5.jpg")

features = extract_frequency_features(img)
cv2dct_features = extract_frequency_features_cv2dct(img)

print("SciPy DCT feature vector:", list(features.values()))
print("cv2.dct feature vector:", list(cv2dct_features.values()))
print("Feature names:", list(features.keys()))
print("Length:", len(features))
print("Close match:", np.allclose(list(features.values()), list(cv2dct_features.values()), rtol=1e-5, atol=1e-7))
