import sys
import dlib
import cv2
import face_recognition
import os
import psycopg2  # Use psycopg2 instead of postgresql
import time
from datetime import datetime
import uuid
import logging
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Database connection configuration
try:
    db_conn = psycopg2.connect(
        user='tapasmohanty',
        password='postgres',
        host='localhost',
        port=5432,
        database='face'
    )
    db_cursor = db_conn.cursor()
    logging.info("Successfully connected to the database.")
except Exception as e:
    logging.error("Failed to connect to the database: {}".format(e))
    sys.exit(1)

# Create faces directory if it doesn't exist
if not os.path.exists("./.faces"):
    os.mkdir("./.faces")
    logging.info("Created directory to store faces: ./.faces")

filename = str(uuid.uuid1())
video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    logging.error("Unable to open the camera")
    sys.exit(1)

# Create HOG face detector using dlib
face_detector = dlib.get_frontal_face_detector()

# Process every nth frame to save resources
frame_skip = 5
frame_count = 0

try:
    while True:
        ret, frame = video_capture.read()

        if not ret:
            logging.error("Failed to capture frame from camera. Exiting...")
            break

        # Process every nth frame
        if frame_count % frame_skip == 0:
            start_time = datetime.now()

            # Detect faces in the frame
            detected_faces = face_detector(frame, 1)
            logging.info("Found {} face(s) in the frame.".format(len(detected_faces)))

            # Loop through each detected face
            for i, face_rect in enumerate(detected_faces):
                logging.info("Processing face #{} at Left: {} Top: {} Right: {} Bottom: {}".format(
                    i, face_rect.left(), face_rect.top(), face_rect.right(), face_rect.bottom()))

                # Crop the face from the frame
                crop = frame[face_rect.top():face_rect.bottom(), face_rect.left():face_rect.right()]

                # Detect face encodings
                encodings = face_recognition.face_encodings(frame, [(face_rect.top(), face_rect.right(),
                                                                     face_rect.bottom(), face_rect.left())])

                if len(encodings) > 0:
                    # Prepare query to insert into the database
                    query = """
                        INSERT INTO vectors (file, vec_low, vec_high)
                        VALUES (%s, CUBE(array[%s]), CUBE(array[%s]))
                    """
                    values = (
                        filename,
                        ','.join(str(s) for s in encodings[0][0:64]),
                        ','.join(str(s) for s in encodings[0][64:128]),
                    )
                    try:
                        # Execute the query
                        db_cursor.execute(query, values)
                        db_conn.commit()
                        logging.info("Inserted face encoding into the database for face #{}".format(i))
                    except Exception as db_error:
                        logging.error("Database insert failed: {}".format(db_error))
                        logging.error(traceback.format_exc())
                        db_conn.rollback()  # Rollback in case of failure

                # Draw a rectangle around the face
                cv2.rectangle(frame, (face_rect.left(), face_rect.top()), (face_rect.right(), face_rect.bottom()), (0, 0, 255), 2)

                # Save cropped face image
                cropped_face_path = "./.faces/aligned_face_{}_{}_crop.jpg".format(filename.replace('/', '_'), i)
                cv2.imwrite(cropped_face_path, crop)
                logging.info("Saved cropped face image to {}".format(cropped_face_path))

            # Display the video with rectangles drawn
            cv2.imshow('Video', frame)

        # Increment the frame counter
        frame_count += 1

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            logging.info("Exiting the video stream.")
            break

except Exception as e:
    logging.error("An error occurred: {}".format(e))
    logging.error(traceback.format_exc())

finally:
    # Clean up resources
    logging.info("Releasing video capture and closing all windows.")
    video_capture.release()
    cv2.destroyAllWindows()

    # Close the database connection
    try:
        db_cursor.close()
        db_conn.close()
        logging.info("Database connection closed.")
    except Exception as db_close_error:
        logging.error("Failed to close the database connection: {}".format(db_close_error))
