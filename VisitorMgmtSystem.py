from flask import Flask, render_template, request, jsonify
import cv2 as cv
import numpy as np
from pymongo import MongoClient
from datetime import datetime

app = Flask(__name__)

# MongoDB connection
client = MongoClient('mongodb://localhost:27017')
db = client['Vms']
visitors_collection = db['vms']

people = ['Aakash G', 'Bevincent Edward E', 'Deva']
DIR = r'D:\vms'
haar_cascade = cv.CascadeClassifier('haar.xml')
face_recognizer = cv.face.LBPHFaceRecognizer_create()

# Load the pre-trained LBPH model
face_recognizer.read('face_trained.yml')

def recognize_face(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    if len(faces_rect) == 0:
        return "No face detected"

    for (x, y, w, h) in faces_rect:
        faces_roi = gray[y:y + h, x:x + w]
        label, confidence = face_recognizer.predict(faces_roi)

        if confidence > 50:
            return "Imposter"
        else:
            return people[label]

@app.route('/')
def index():
    return render_template('vmsfront.html')


@app.route('/recognize', methods=['POST'])
def recognize():
    file = request.files['image']
    img = cv.imdecode(np.fromstring(file.read(), np.uint8), cv.IMREAD_COLOR)

    recognized_person = recognize_face(img)

    visitor_data = {
        'name': recognized_person,
        'time_checked_in': datetime.now()
    }
    visitors_collection.insert_one(visitor_data)

    return jsonify({'recognized_person': recognized_person})


if __name__ == '__main__':
    app.run(debug=True)