# recognize_face.py
import cv2
import dlib
import face_recognition
import psycopg2
import logging
from dotenv import load_dotenv
import os
import sys

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Database connection
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

def match_face(encoding):
    """Compare face encoding with database records"""
    query = "SELECT name, vec_low, vec_high FROM vectors"
    db_cursor.execute(query)
    rows = db_cursor.fetchall()

    for row in rows:
        name, vec_low, vec_high = row
        db_encoding = vec_low + vec_high  # Combine the two halves of the encoding
        matches = face_recognition.compare_faces([db_encoding], encoding)
        if matches[0]:
            return name
    return "Unknown"


def recognize_faces():
    video_capture = cv2.VideoCapture(0)
    face_detector = dlib.get_frontal_face_detector()

    while True:
        ret, frame = video_capture.read()
        if not ret:
            logging.error("Failed to capture frame from camera.")
            break

        # Detect faces
        detected_faces = face_detector(frame, 1)

        # Process detected faces
        for face_rect in detected_faces:
            face_encodings = face_recognition.face_encodings(frame, [(face_rect.top(), face_rect.right(), face_rect.bottom(), face_rect.left())])

            if len(face_encodings) > 0:
                recognized_name = match_face(face_encodings[0])
                logging.info(f"Recognized: {recognized_name}")

                # Draw a rectangle and label around the face
                cv2.rectangle(frame, (face_rect.left(), face_rect.top()), (face_rect.right(), face_rect.bottom()), (0, 255, 0), 2)
                cv2.putText(frame, recognized_name, (face_rect.left(), face_rect.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Show the frame with the results
        cv2.imshow('Face Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    recognize_faces()

    # Close the database connection
    try:
        db_cursor.close()
        db_conn.close()
        logging.info("Database connection closed.")
    except Exception as db_close_error:
        logging.error("Failed to close the database connection: {}".format(db_close_error))
