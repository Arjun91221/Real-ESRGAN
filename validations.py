import tensorflow as tf
import os
import cv2
import math
import matplotlib.pyplot as plt
import numpy as np
import dlib

current_directory = os.path.dirname(__file__)

model_dir = f"{current_directory}/train_faceclass"

loaded_model = tf.keras.models.load_model(model_dir)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

predictor_model_dir = f"{current_directory}/models/shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(predictor_model_dir)


async def checkfacestraight(image_path):
    image = cv2.imread(image_path)

    # Convert the image to grayscale for face detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(30, 30))

    print(f"faces {faces}")

    # Check the number of detected faces
    if len(faces) > 1:
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=20, minSize=(30, 30))

    if len(faces) > 1:
        return "Multiple faces detected."

    # Check if faces are found
    if len(faces) == 0:
        return "Face is not clear."

    # Define acceptable tilt angle range (in degrees)
    acceptable_tilt_angle = 90 # Adjust this value as needed

    # Define an acceptable eye angle range (adjust this range as needed)
    acceptable_eye_angle_range = (-8, -5)  # For example, 0 to 10 degrees

    # Define standard passport photo dimensions
    passport_width = 35  # Width in millimeters
    passport_height = 45  # Height in millimeters

    # Calculate the aspect ratio of the passport photo
    aspect_ratio = passport_width / passport_height

    # Calculate the width and height of the passport-sized image in pixels
    new_width = 250  # You can adjust this width as needed
    new_height = int(new_width / aspect_ratio)


    # Process the detected face
    for (x, y, w, h) in faces:

        # Crop the face region
        face = image[y:y + h, x:x + w]

        face = cv2.resize(face, (224, 224))

        class_names = ['notstraight', 'straight']

        img_array = tf.keras.utils.img_to_array(face)
        img_array = tf.expand_dims(img_array, 0)  # Create a batch

        predictions = loaded_model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        predicted_class = class_names[np.argmax(score)]
        confidence = 100 * np.max(score)


        # Commented out for straight face
        # if predicted_class != 'straight':
        #     return "The face is not straight. Please, upload a straight human face"


        # Convert the OpenCV rectangle to a Dlib rectangle
        dlib_rect = dlib.rectangle(x, y, x + w, y + h)

        # Detect facial landmarks within the face rectangle
        landmarks = predictor(gray, dlib_rect)


        # Loop through the facial landmarks and draw circles at each landmark point
        for i in range(68):  # There are 68 facial landmarks in dlib's 68_face_landmarks.dat
            xl, yl = landmarks.part(i).x, landmarks.part(i).y
            #cv2.circle(image, (xl, yl), 2, (0, 0, 255), -1)  # Draw a red circle at each landmark point

        # Extract specific landmarks (e.g., left eye, right eye, and chin)
        nose = landmarks.part(30)
        left_eye = landmarks.part(36)
        right_eye = landmarks.part(45)
        chin = landmarks.part(8)

        # Calculate the angle between the eyes
        eye_angle = calculate_angle(left_eye, right_eye)

        # Calculate the angle between the eyes and chin landmarks
        eye_chin_angle = calculate_angle(left_eye, chin)

        #print(eye_angle)
        #print(eye_chin_angle)

        # Check if the face is tilted within the acceptable range
        # if abs(eye_chin_angle) > acceptable_tilt_angle:
        #     return "Please, upload a straight human face"

        # Determine if the face is straight based on the eye angle
        # if acceptable_eye_angle_range[0] <= eye_angle <= acceptable_eye_angle_range[1]:
        #     return "Please, upload a straight human face"
        # else:
        return "The face is straight."


        # Expand the cropping region to include the hair, cap, or turban
        expand_factor = 2.0  # Adjust this factor as needed
        expanded_x = max(0, int(x - 0.5 * w * (expand_factor - 1)))
        expanded_y = max(0, int(y - 0.5 * h * (expand_factor - 1)))
        expanded_width = min(image.shape[1] - expanded_x, int(w * expand_factor))
        expanded_height = [min(image.shape[0] - expanded_y, int(h * (expand_factor-0.5))), min(image.shape[0] - expanded_y, int(h * (expand_factor-0.3))), min(image.shape[0] - expanded_y, int(h * (expand_factor-0.2))), min(image.shape[0] - expanded_y, int(h * expand_factor)), min(image.shape[0] - expanded_y, int(h * (expand_factor+0.2))), min(image.shape[0] - expanded_y, int(h * (expand_factor+0.3))), min(image.shape[0] - expanded_y, int(h * (expand_factor+0.5))), min(image.shape[0] - expanded_y, int(h * (expand_factor+0.75))), min(image.shape[0] - expanded_y, int(h * (expand_factor+1))), min(image.shape[0] - expanded_y, int(h * (expand_factor+1.25)))]

        # Calculate the expansion on both sides
        #expand_x = int((expanded_width - w) / 2)
        # Apply the expansion equally on both sides
        #expanded_x = max(0, int(x - expand_x))
        # Verify that the expanded region stays within the image boundaries
        #expanded_x = min(expanded_x - 50 , image.shape[1] - expanded_width)

        extra_width = [0, 10, 15, 20, 30, 35, 40, 50, 50, 55]
        for i in range(len(expanded_height)):
            # Crop the expanded region
            cropped = image[expanded_y:expanded_y + expanded_height[i], expanded_x - extra_width[i] :expanded_x + expanded_width + extra_width[i]]

            #passport_size_image = cv2.resize(cropped, (new_width, new_height))
            cv2.imwrite("/content/cropped{}.jpg".format(i), cropped)

        # Draw annotations on the image
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw a green rectangle around the face
        cv2.putText(image, "Eye-Chin Angle: {:.2f}".format(eye_chin_angle), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(image, "Eye Angle: {:.2f}".format(eye_angle), (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)



def calculate_angle(point1, point2):
    x1, y1 = point1.x, point1.y
    x2, y2 = point2.x, point2.y
    return math.degrees(math.atan2(y2 - y1, x2 - x1))

# def loadImg(img_path):
#     img = face_recognition.load_image_file(img_path)
#     return img


# def getFaceLoc(img):
#     face_loc = face_recognition.face_locations(img)
#     if len(face_loc)==0:
#         return None

#     return face_loc


# def extFace(face_loc, img):
#     top, right, bottom, left = face_loc[0]

#     extracted_face = img[top:bottom, left:right]
#     return extracted_face

# import cv2
# import numpy as np
# import torch
# from facenet_pytorch import InceptionResnetV1
# from scipy.spatial.distance import cosine

# def faceMatch(image1,image2):
#     model = InceptionResnetV1(pretrained='vggface2')
#     model.eval()

#     image1_tensor = torch.from_numpy(image1).permute(2, 0, 1).unsqueeze(0).float()
#     image2_tensor = torch.from_numpy(image2).permute(2, 0, 1).unsqueeze(0).float()

#     with torch.no_grad():
#         embedding1 = model(image1_tensor)
#         embedding2 = model(image2_tensor)

#     embedding1_flat = embedding1.numpy().flatten()
#     embedding2_flat = embedding2.numpy().flatten()

#     similarity_score = 1 - cosine(embedding1_flat, embedding2_flat)

#     threshold = 0.8

#     print(similarity_score)
#     if similarity_score >= threshold:
#         return True

#     return False

# def checkFaceSimilarity(img1_path, img2_path):
#     image_1 = face_recognition.load_image_file(img1_path)
#     image_1_encoding = face_recognition.face_encodings(image_1)[0]

#     image_2 = face_recognition.load_image_file(img2_path)
#     image_2_encoding = face_recognition.face_encodings(image_2)[0]

#     threshold = 0.57
#     distance = face_recognition.face_distance([image_1_encoding], image_2_encoding)[0]
#     print(f"Distance: {distance:.2f} (threshold {threshold:.2f})")


#     if distance < threshold:
#         return True

#     return False

