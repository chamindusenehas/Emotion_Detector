import cv2
import numpy as np
from tensorflow.keras.models import load_model


model = load_model('detection_model.h5')

emotion_labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'surprise', 6: 'neutral'}

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
       
        face = frame[y:y+h, x:x+w] 
        face = cv2.resize(face, (224, 224))
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face_rgb = face_rgb / 255.0
        face_rgb = np.expand_dims(face_rgb, axis=0)

        pred = model.predict(face_rgb)
        emotion_idx = np.argmax(pred)
        emotion = emotion_labels[emotion_idx]

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('Emotion Recognizer', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()