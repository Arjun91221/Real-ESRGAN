import cv2
import numpy as np
import torch
from scipy.spatial.distance import cosine
import face_recognition
import onnxruntime as rt
import mediapipe as mp
from types import SimpleNamespace
from typing import List
import numpy as np
import cv2


FACE_DETECTOR = mp.solutions.face_mesh.FaceMesh(
    refine_landmarks=True, min_detection_confidence=0.7, min_tracking_confidence=0.6, max_num_faces=100
)

async def checkFaceSimilarity(img1_path, img2_path):
    image_1 = face_recognition.load_image_file(img1_path)
    image_1_encoding = face_recognition.face_encodings(image_1)[0]

    image_2 = face_recognition.load_image_file(img2_path)
    image_2_encoding = face_recognition.face_encodings(image_2)[0]

    threshold = 0.61
    distance = face_recognition.face_distance([image_1_encoding], image_2_encoding)[0]
    print(f"Distance: {distance:.2f} (threshold {threshold:.2f})")


    if distance < threshold:
        return True

    return False


def loadImg(img_path):
    img = face_recognition.load_image_file(img_path)
    return img


def getFaceLoc(img):
    face_loc = face_recognition.face_locations(img)
    if len(face_loc)==0:
        return None

    return face_loc


def extFace(face_loc, img):
    top, right, bottom, left = face_loc[0]

    extracted_face = img[top:bottom, left:right]
    return extracted_face



# Define a class to store a detection
class Detection(SimpleNamespace):
    bbox: List[List[float]] = None
    landmarks: List[List[float]] = None


def detect_faces(frame: np.ndarray) -> List[Detection]:
    # Process the frame with the face detector
    result = FACE_DETECTOR.process(frame)
    # Initialize an empty list to store the detected faces
    detections = []

    # Check if any faces were detected
    if result.multi_face_landmarks:
        # Iterate over each detected face
        for count, detection in enumerate(result.multi_face_landmarks):
            # Select 5 Landmarks
            five_landmarks = np.asarray(detection.landmark)[[470, 475, 1, 57, 287]]

            # Extract the x and y coordinates of the landmarks of interest
            landmarks = np.asarray(
                [[landmark.x * frame.shape[1], landmark.y * frame.shape[0]] for landmark in five_landmarks]
            )

            # Extract the x and y coordinates of all landmarks
            all_x_coords = [landmark.x * frame.shape[1] for landmark in detection.landmark]
            all_y_coords = [landmark.y * frame.shape[0] for landmark in detection.landmark]

            # Compute the bounding box of the face
            x_min, x_max = int(min(all_x_coords)), int(max(all_x_coords))
            y_min, y_max = int(min(all_y_coords)), int(max(all_y_coords))
            bbox = [[x_min, y_min], [x_max, y_max]]

            # Create a Detection object for the face
            detection = Detection(idx=count, bbox=bbox, landmarks=landmarks, confidence=None)

            # Add the detection to the list
            detections.append(detection)

    # Return the list of detections
    return detections


async def checkfacevalidation(image_path):
    #run detection
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detections = detect_faces(image_rgb)

    if len(detections) > 1:
        print("multiple Faces Detected")
        return False , "Multiple Faces Detected"
    elif len(detections) == 1:
        print("Single Human face Detected")
        return True , "Single Human face Detected"
    else:
        print("Human Face not Found")
        return False , "Human Face not Found"
