import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

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

# =========================
# 2. Load image
# =========================
image_path = "/home/testuser/Desktop/MediaPipeDeepFakeDetectionPrototype/FacialLandmarkData/test1.jpg"

mp_image = mp.Image.create_from_file(image_path)
opencv_image = cv2.imread(image_path)

# =========================
# 3. Detect landmarks
# =========================
result = detector.detect(mp_image)

# =========================
# 4. Draw landmarks
# =========================
if result.face_landmarks:

    h, w, _ = opencv_image.shape

    print(f"Detected {len(result.face_landmarks)} face(s)")

    for face_index, face_landmarks in enumerate(result.face_landmarks):

        print(f"\nFace {face_index}:")

        # -------------------------
        # Print ONLY the first 10
        # -------------------------
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

        # -------------------------
        # Draw ALL landmarks
        # -------------------------
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
    print("No face detected.")

# =========================
# 5. Show result
# =========================
cv2.imshow("Face Landmarks", opencv_image)
cv2.waitKey(0)
cv2.destroyAllWindows()