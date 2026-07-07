import numpy as np

LEFT_EYE = 33
RIGHT_EYE = 263
NOSE = 1

def normalize_landmarks(landmarks):
    """
    Normalize MediaPipe landmarks by removing translation and scale.

    Returns:
        numpy array of shape (478,3)
    """

    points = np.array(
        [[lm.x, lm.y, lm.z] for lm in landmarks],
        dtype=np.float32
    )

    # Move nose to origin
    points -= points[NOSE]

    # Scale by eye distance
    eye_distance = np.linalg.norm(
        points[LEFT_EYE, :2] - points[RIGHT_EYE, :2]
    )

    if eye_distance > 1e-6:
        points /= eye_distance

    return points