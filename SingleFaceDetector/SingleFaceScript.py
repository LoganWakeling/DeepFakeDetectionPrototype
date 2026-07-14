import os
import shutil
import cv2

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp

# -----------------------------
# Configuration
# -----------------------------
# make input path the folder which has the images you want to sort through
# make output path the folder which you wanted all single face images to be outputted to

INPUT_FOLDER = "/home/vboxuser/Desktop/DeepFakeResearchProject/WIDER_train/images/52--Photographers"
OUTPUT_FOLDER = "/home/vboxuser/Desktop/DeepFakeResearchProject/WIDER_train/collection"
MODEL_PATH = "face_landmarker.task"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# -----------------------------
# Create detector
# -----------------------------
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=5  # allow detection of multiple faces
)

detector = vision.FaceLandmarker.create_from_options(options)

# -----------------------------
# Supported image types
# -----------------------------
extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

moved = 0
skipped = 0

# -----------------------------
# Process images
# -----------------------------
for filename in os.listdir(INPUT_FOLDER):

    if not filename.lower().endswith(extensions):
        continue

    filepath = os.path.join(INPUT_FOLDER, filename)

    image = cv2.imread(filepath)

    if image is None:
        print(f"Couldn't read {filename}")
        continue

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = detector.detect(mp_image)

    num_faces = len(result.face_landmarks)

    if num_faces == 1:
        destination = os.path.join(OUTPUT_FOLDER, filename)

        # Move (cut & paste)
        shutil.move(filepath, destination)

        moved += 1
        print(f"Moved: {filename}")

    else:
        skipped += 1
        print(f"Skipped ({num_faces} faces): {filename}")

print()
print(f"Moved: {moved}")
print(f"Skipped: {skipped}")
