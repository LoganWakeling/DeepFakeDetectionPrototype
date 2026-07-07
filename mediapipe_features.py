import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import facial_regions
import numpy as np
import glob
import os
from normalization import normalize_landmarks


# =========================
# 1. Create the landmarker
# =========================
base_options = python.BaseOptions(
    model_asset_path="face_landmarker.task"
)

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True,
    num_faces=1
)

detector = vision.FaceLandmarker.create_from_options(options)


# ===============================================
# getting landmarks and creating image of them
# ===============================================
def process_image(image_path):

    # Load image
    mp_image = mp.Image.create_from_file(image_path)
    opencv_image = cv2.imread(image_path)

    # Detect landmarks
    result = detector.detect(mp_image)

    if result.face_landmarks:

        h, w, _ = opencv_image.shape


        for face_index, face_landmarks in enumerate(result.face_landmarks):


            # Draw all landmarks
            for landmark in face_landmarks:

                x = int(landmark.x * w)
                y = int(landmark.y * h)

                cv2.circle(
                    opencv_image,
                    (x, y),
                    1,
                    (0, 255, 0),
                    -1
                )

    else:
        print(f"No face detected in {image_path}")

    # Show image
    cv2.imshow(image_path, opencv_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



# ==========================================
# extract all landmarks into an array
# ==========================================
def extract_landmarks(image_path):
    mp_image = mp.Image.create_from_file(image_path)
    result = detector.detect(mp_image)

    # Check if a face was detected
    if not result.face_landmarks:
        return None

    # Get first detected face
    face_landmarks = result.face_landmarks[0]

    # Normalize the landmarks
    landmarks = normalize_landmarks(face_landmarks)

    return landmarks



# ========================================
# split up landmark region points
# ========================================
def split_landmarks_by_region(landmarks):
    return {
        name: landmarks[indices]
        for name, indices in facial_regions.FACIAL_REGIONS.items()
    }



def compare_regions(regions1, regions2):
    """
    Compare two faces by calculating the average landmark distance
    for each facial region.

    Returns:
        scores (dict): Average distance for each facial region.
        overall_score (float): Mean distance across all regions.
    """

    scores = {}

    for region in facial_regions.FACIAL_REGIONS:

        pts1 = np.asarray(regions1[region])
        pts2 = np.asarray(regions2[region])

        # Skip if the regions don't contain the same number of landmarks
        if pts1.shape != pts2.shape:
            scores[region] = None
            continue

        # Euclidean distance for each landmark pair
        distances = np.linalg.norm(pts1 - pts2, axis=1)

        # Mean distance for this region
        scores[region] = np.mean(distances)

    # Compute an overall similarity score
    valid_scores = [s for s in scores.values() if s is not None]
    overall_score = np.mean(valid_scores) if valid_scores else None

    return scores, overall_score



# =========================
# define FACIAL_REGIONS
# =========================
FACIAL_REGIONS = facial_regions.FACIAL_REGIONS


# ========================================================
# import image from database 
# Extracting landmarks and making image representation
# ========================================================

def run_landmark_extraction(file_path):
    image_path = file_path
    print(image_path)
    process_image(image_path)
    image1 = extract_landmarks(image_path)

    if image1 is None:
        print("No face detected.")
        return

    region1 = split_landmarks_by_region(image1)

    print("ladmarks extracted")
    return region1

    

# ================
# Testing
# ================

#photo paths and extracting landmarks into seperate regions (eyes, nose, mouth, etc.)
path1 = "/home/testuser/Desktop/MediaPipeDeepFakeDetectionPrototype/FacialLandmarkData/obama1.jpg"
path2 = "/home/testuser/Desktop/MediaPipeDeepFakeDetectionPrototype/FacialLandmarkData/obama5.jpg"
region1 = run_landmark_extraction(path1)
region2 = run_landmark_extraction(path2)

#use this when comparing two photos to get the landmark differences 
scores, overall = compare_regions(region1, region2)

print("Overall Score:", overall)

for region, score in scores.items():
    print(f"{region:15}: {score:.5f}")









