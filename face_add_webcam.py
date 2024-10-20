import sys
import dlib
import cv2
import face_recognition
import os
import psycopg2
import uuid
import logging
import traceback
from tkinter import Tk, Label, Entry, Button, StringVar
from dotenv import load_dotenv  # Import dotenv to load environment variables
import os  # Import os to access environment variables

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Database connection configuration using environment variables
try:
    db_conn = psycopg2.connect(
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        database=os.getenv("DB_NAME")
    )
    db_cursor = db_conn.cursor()
    logging.info("Successfully connected to the database.")
except Exception as e:
    logging.error("Failed to connect to the database: {}".format(e))
    sys.exit(1)

# Create directory for saving face images
if not os.path.exists("./.faces"):
    os.mkdir("./.faces")
    logging.info("Created directory to store faces: ./.faces")


def start_face_registration():
    person_name = name_var.get()
    if not person_name:
        result_label.config(text="Please enter a name")
        return
    
    # Start capturing 5 images of the person
    capture_images(person_name)


def capture_images(person_name):
    video_capture = cv2.VideoCapture(0)
    face_detector = dlib.get_frontal_face_detector()

    captured_faces = 0
    frame_count = 0
    frame_skip = 5
    encodings_list = []

    while captured_faces < 5:
        ret, frame = video_capture.read()

        if not ret:
            logging.error("Failed to capture frame from camera.")
            break

        # Process every nth frame
        if frame_count % frame_skip == 0:
            # Detect faces
            detected_faces = face_detector(frame, 1)

            for i, face_rect in enumerate(detected_faces):
                if captured_faces >= 5:
                    break

                logging.info(f"Processing face at Left: {face_rect.left()}, Top: {face_rect.top()}, Right: {face_rect.right()}, Bottom: {face_rect.bottom()}")

                # Crop the face
                crop = frame[face_rect.top():face_rect.bottom(), face_rect.left():face_rect.right()]

                # Save the image
                filename = f"./.faces/{person_name}_{captured_faces}.jpg"
                cv2.imwrite(filename, crop)
                logging.info(f"Saved cropped face image to {filename}")

                # Detect face encodings
                encodings = face_recognition.face_encodings(frame, [(face_rect.top(), face_rect.right(), face_rect.bottom(), face_rect.left())])

                if len(encodings) > 0:
                    encodings_list.append(encodings[0].tolist())
                    captured_faces += 1

            # Display the frame with rectangles around the face
            for face_rect in detected_faces:
                cv2.rectangle(frame, (face_rect.left(), face_rect.top()), (face_rect.right(), face_rect.bottom()), (0, 0, 255), 2)
            cv2.imshow('Capturing Faces', frame)

        # Increment frame counter
        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

    if captured_faces == 5:
        store_face_data(person_name, encodings_list)
        result_label.config(text=f"Registered face for {person_name}")
    else:
        result_label.config(text="Failed to capture sufficient images")


def store_face_data(person_name, encodings_list):
    """Store face encodings and name into the database"""
    try:
        for encoding in encodings_list:
            query = """
                INSERT INTO vectors (file, vec_low, vec_high, name)
                VALUES (%s, CUBE(%s::double precision[]), CUBE(%s::double precision[]), %s)
            """
            values = (
                person_name,  # Using the name as filename in this case
                encoding[0:64],
                encoding[64:128],
                person_name
            )
            db_cursor.execute(query, values)
        db_conn.commit()
        logging.info(f"Stored face encodings for {person_name} in database")
    except Exception as e:
        logging.error("Failed to store face encodings: {}".format(e))
        db_conn.rollback()


# Tkinter GUI for user input
app = Tk()
app.title("Face Registration")

name_var = StringVar()

Label(app, text="Enter name: ").grid(row=0, column=0)
Entry(app, textvariable=name_var).grid(row=0, column=1)
Button(app, text="Start Face Registration", command=start_face_registration).grid(row=1, column=1)
result_label = Label(app, text="")
result_label.grid(row=2, column=1)

app.mainloop()

# Close the database connection
try:
    db_cursor.close()
    db_conn.close()
    logging.info("Database connection closed.")
except Exception as db_close_error:
    logging.error("Failed to close the database connection: {}".format(db_close_error))
