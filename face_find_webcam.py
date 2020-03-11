import face_recognition
import cv2
import dlib
import psycopg2
import time
from datetime import datetime
from datetime import timedelta

video_capture = cv2.VideoCapture(0)

connection_db = psycopg2.connect(
"user='postgres' password='5432' host='localhost' dbname='face' password = 'postgres'")
db = connection_db.cursor()

while True:
	# Grab a single frame of video
	ret, frame = video_capture.read()

	face_detector = dlib.get_frontal_face_detector()
	detected_faces = face_detector(frame, 1)

	# Loop through each face in this frame of video
	start_time = datetime.now()
	for i, face_rect in enumerate(detected_faces):
		crop = frame[face_rect.top():face_rect.bottom(), face_rect.left():face_rect.right()]
		encodings = face_recognition.face_encodings(crop)

		threshold = 0.4
		
		if len(encodings) > 0:
			query = "SELECT file FROM vectors WHERE sqrt(power(CUBE(array[{}]) <-> vec_low, 2) + power(CUBE(array[{}]) <-> vec_high, 2)) <= {} ".format(
				','.join(str(s) for s in encodings[0][0:64]),
				','.join(str(s) for s in encodings[0][64:128]),
				threshold,
			) + \
				"ORDER BY sqrt((CUBE(array[{}]) <-> vec_low) + (CUBE(array[{}]) <-> vec_high)) ASC LIMIT 5".format(
				','.join(str(s) for s in encodings[0][0:64]),
				','.join(str(s) for s in encodings[0][64:128]),
			)
			db.execute(query)

			name = db.fetchone()

			dt = datetime.now() - start_time
			ms = (dt.days * 24 * 60 * 60 + dt.seconds) * \
				1000 + dt.microseconds / 1000.0            

			# Draw a box around the face
			cv2.rectangle(frame, (face_rect.left(), face_rect.top()), (face_rect.right(), face_rect.bottom()), (0, 0, 255), 2)
			# Draw a label with a name below the face
			cv2.rectangle(frame, (face_rect.left(), face_rect.bottom() - 35), (face_rect.right(), face_rect.bottom()), (0, 0, 255), cv2.FILLED)
			cv2.putText(frame, "Name: {}, Time {}".format(name, ms), (int(face_rect.left()), int(face_rect.bottom())-15), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

		else:
			print("No encodings")
	# Display the resulting image
	cv2.imshow('Video', frame)

	# Hit 'q' on the keyboard to quit!
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
		
video_capture.release()
cv2.destroyAllWindows()