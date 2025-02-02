from flask import Flask, render_template, request, jsonify
import cv2
import tensorflow as tf
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__, template_folder="templates")
app.config["UPLOAD_FOLDER"] = "uploads/"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Load AI model
MODEL_PATH = "resnet50_fight_detection_finetuned.h5"
model = tf.keras.models.load_model(MODEL_PATH)

def process_video(file_path):
    """Extracts frames from video and predicts if it's a fight or non-fight."""
    cap = cv2.VideoCapture(file_path)
    predictions = []

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break  # Stop if the video ends

        # Preprocess frame for the AI model
        img = cv2.resize(frame, (224, 224))
        img_array = np.expand_dims(img / 255.0, axis=0)  # Normalize

        # Make prediction
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)
        predictions.append(predicted_class)

    cap.release()

    # Final decision: if majority frames indicate fight, return "Fight detected"
    fight_frames = predictions.count(1)
    non_fight_frames = predictions.count(0)

    return "Fight detected!" if fight_frames > non_fight_frames else "No fight detected."

@app.route('/incidents')
def incidents():
    return render_template("incidents.html")

@app.route('/')
def home():
    return render_template("main.html")

@app.route('/predict', methods=["POST"])
def predict():
    """Handle video upload and return AI prediction."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)

    # Process the uploaded video
    prediction = process_video(file_path)

    return jsonify({"prediction": prediction})

if __name__ == "__main__":
    app.run(debug=True)

