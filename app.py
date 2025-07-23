# app.py
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
model = load_model(r"C:\Users\User\Desktop\Project Frame\gait_model.keras")  # now a .keras single file
  # Load SavedModel
le = LabelEncoder()
le.fit(['113', '114', '115', '116', '117', '118', '119'])

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return jsonify({'error': 'Invalid image'}), 400

    img = cv2.resize(img, (128, 128))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=(0, -1))  # shape: (1, 128, 128, 1)

    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)
    confidence = float(np.max(prediction))
    subject = le.inverse_transform([predicted_class])[0]

    return jsonify({'predicted_subject': subject, 'confidence': confidence})

if __name__ == '__main__':
    app.run(debug=True)
