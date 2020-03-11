import sys
import dlib
import cv2
import face_recognition
import os
import postgresql
import postgresql.driver as pg_driver
import time
from datetime import datetime
from datetime import timedelta
import uuid

filename = str(uuid.uuid1())
video_capture = cv2.VideoCapture(0)

if not os.path.exists("./.faces"):
    os.mkdir("./.faces")

db = pg_driver.connect(user = 'postgres',password = 'postgres', host = 'localhost', port = 5432, database = 'face')

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()
    # Create a HOG face detector using the built-in dlib class
    face_detector = dlib.get_frontal_face_detector()

    # Run the HOG face detector on the image data
    detected_faces = face_detector(frame, 1)
   
    print("Found {} faces in the image file {}".format(len(detected_faces), filename))

    # Loop through each face in this frame of video
    start_time = datetime.now()
    # Loop through each face we found in the image
    for i, face_rect in enumerate(detected_faces):
        # Detected faces are returned as an object with the coordinates
        # of the top, left, right and bottom edges
        print("- Face #{} found at Left: {} Top: {} Right: {} Bottom: {}".format(i, face_rect.left(), face_rect.top(),
                                                                                 face_rect.right(), face_rect.bottom()))
        crop = frame[face_rect.top():face_rect.bottom(), face_rect.left():face_rect.right()]
        encodings = face_recognition.face_encodings(crop)

        if len(encodings) > 0:
            query = "INSERT INTO vectors (file, vec_low, vec_high) VALUES ('{}', CUBE(array[{}]), CUBE(array[{}]))".format(
                filename,
                ','.join(str(s) for s in encodings[0][0:64]),
                ','.join(str(s) for s in encodings[0][64:128]),
            )
            db.execute(query)
        # Draw a box around the face
        cv2.rectangle(frame, (face_rect.left(), face_rect.top()), (face_rect.right(), face_rect.bottom()), (0, 0, 255), 2)
        cv2.imwrite("./.faces/aligned_face_{}_{}_crop.jpg".format(filename.replace('/', '_'), i), crop)
    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
video_capture.release()
cv2.destroyAllWindows()
