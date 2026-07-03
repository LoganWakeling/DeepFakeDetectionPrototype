import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import facial_regions
import numpy as np

print("Facial Regions working")

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


def process_image(image_path):

    # Load image
    mp_image = mp.Image.create_from_file(image_path)
    opencv_image = cv2.imread(image_path)

    # Detect landmarks
    result = detector.detect(mp_image)

    if result.face_landmarks:

        h, w, _ = opencv_image.shape

        print(f"\nProcessing: {image_path}")
        print(f"Detected {len(result.face_landmarks)} face(s)")

        for face_index, face_landmarks in enumerate(result.face_landmarks):

            print(f"\nFace {face_index}:")

            # Print first 10 landmarks
            for landmark_index, landmark in enumerate(face_landmarks[:10]):

                x = int(landmark.x * w)
                y = int(landmark.y * h)

                print(
                    f"Landmark {landmark_index}: "
                    f"normalized=({landmark.x:.4f}, "
                    f"{landmark.y:.4f}, "
                    f"{landmark.z:.4f}) "
                    f"pixels=({x}, {y})"
                )

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



def extract_landmarks(image_path):

    mp_image = mp.Image.create_from_file(image_path)
    result = detector.detect(mp_image)

    if not result.face_landmarks:
        return None

    # Get first detected face
    face_landmarks = result.face_landmarks[0]

    # Store normalized coordinates
    landmarks = np.array([
        [lm.x, lm.y, lm.z]
        for lm in face_landmarks
    ])

    return landmarks


# =========================
# Paths of two images
# =========================
image_paths = [
    "/home/testuser/Desktop/MediaPipeDeepFakeDetectionPrototype/FacialLandmarkData/test2.jpg",
    "/home/testuser/Desktop/MediaPipeDeepFakeDetectionPrototype/FacialLandmarkData/test3.jpg"
]

# ==========================
#Extracting Landmarks
# ==========================
image1 = extract_landmarks(image_paths[0])
image2 = extract_landmarks(image_paths[1])


# ========================================
#Print out image with landmarks on it
# ========================================
for path in image_paths:
    process_image(path)

