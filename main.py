import os
import cv2
import dlib
import face_recognition
import psycopg2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from dotenv import load_dotenv
from typing import List
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Database connection configuration
try:
    db_conn = psycopg2.connect(
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        database=os.getenv("DB_NAME")
    )
    db_cursor = db_conn.cursor()
    print("Successfully connected to the database.")
except Exception as e:
    print(f"Failed to connect to the database: {e}")
    raise HTTPException(status_code=500, detail="Database connection failed")

# Initialize dlib face detector
face_detector = dlib.get_frontal_face_detector()

def send_email_notification(image_path):
    """Send an email notification with the attached image of the unidentified person."""
    sender_email = "your_email@gmail.com"
    receiver_email = "receiver_email@gmail.com"  # Replace with your email
    password = os.getenv("EMAIL_PASSWORD")  # Store your email password in .env

    subject = "Unidentified Person Detected"
    body = "An unidentified person was detected by the ESP32-CAM. Please check the attached image."

    # Create the email message
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    # Attach the image
    with open(image_path, "rb") as attachment:
        part = MIMEBase("application", "octet-stream")
        part.set_payload(attachment.read())
        encoders.encode_base64(part)
        part.add_header(
            "Content-Disposition",
            f"attachment; filename= {os.path.basename(image_path)}",
        )
        msg.attach(part)

    # Send the email
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, password)
        text = msg.as_string()
        server.sendmail(sender_email, receiver_email, text)
        server.quit()
        print("Email notification sent successfully.")
    except Exception as e:
        print(f"Failed to send email: {e}")


def store_face_data(person_name, encodings_list):
    """Store face encodings and name into the database."""
    try:
        for encoding in encodings_list:
            query = """
                INSERT INTO vectors (file, vec_low, vec_high, name)
                VALUES (%s, CUBE(%s::double precision[]), CUBE(%s::double precision[]), %s)
            """
            values = (
                person_name,
                encoding[0:64],
                encoding[64:128],
                person_name
            )
            db_cursor.execute(query, values)
        db_conn.commit()
        print(f"Stored face encodings for {person_name} in database.")
    except Exception as e:
        print(f"Failed to store face encodings: {e}")
        db_conn.rollback()
        raise HTTPException(status_code=500, detail="Database insert failed.")


@app.post("/register_face/")
async def register_face(name: str, file: UploadFile = File(...)):
    """API to register a new face."""
    # Read the image data from the uploaded file
    image_data = await file.read()

    # Convert the byte data to a numpy array
    np_img = np.frombuffer(image_data, np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    # Detect faces
    detected_faces = face_detector(frame, 1)

    if len(detected_faces) == 0:
        raise HTTPException(status_code=400, detail="No faces detected in the image.")

    encodings_list = []
    for face_rect in detected_faces:
        encodings = face_recognition.face_encodings(frame, [(face_rect.top(), face_rect.right(), face_rect.bottom(), face_rect.left())])
        if len(encodings) > 0:
            encodings_list.append(encodings[0].tolist())

    # Store the face data
    store_face_data(name, encodings_list)

    return {"message": f"Successfully registered face for {name}."}


def match_face(encoding):
    """Compare face encoding with database records."""
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


@app.post("/recognize_face/")
async def recognize_face(file: UploadFile = File(...)):
    """API to recognize faces in the uploaded image."""
    # Read the image data from the uploaded file
    image_data = await file.read()

    # Convert the byte data to a numpy array
    np_img = np.frombuffer(image_data, np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    # Detect faces
    detected_faces = face_detector(frame, 1)

    if len(detected_faces) == 0:
        raise HTTPException(status_code=400, detail="No faces detected in the image.")

    recognized_faces = []
    for face_rect in detected_faces:
        encodings = face_recognition.face_encodings(frame, [(face_rect.top(), face_rect.right(), face_rect.bottom(), face_rect.left())])
        if len(encodings) > 0:
            recognized_name = match_face(encodings[0])
            recognized_faces.append(recognized_name)

            # If the person is unidentified, send an email
            if recognized_name == "Unknown":
                print("Unidentified person detected, sending email notification...")
                # Save the image and send the email
                image_path = "./unidentified_person.jpg"
                cv2.imwrite(image_path, frame)
                send_email_notification(image_path)

    return {"recognized_faces": recognized_faces}
