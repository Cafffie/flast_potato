from flask import Flask, request, jsonify, url_for, render_template, send_from_directory
from PIL import Image
import numpy as np
from werkzeug.utils import secure_filename
import tensorflow as tf
import os

app = Flask(__name__)

# Define the upload folder
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')

# Ensure the upload folder exists, if not, create it
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the pre-trained model
MODEL_PATH = r"C:\projects\potato_disease\training\potatoes_plant_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Define class names
CLASS_NAMES = ['Early_blight', 'Late_blight', 'healthy']
IMAGE_SIZE = 256

def predict_image(model, img):
    image_arr = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    image_arr = tf.keras.preprocessing.image.img_to_array(image_arr)
    image_arr = tf.expand_dims(image_arr, 0)

    predictions = model.predict(image_arr)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = round(100 * np.max(predictions[0]), 2)

    return predicted_class, confidence

@app.route("/")
def hello():
    return render_template("index.html")

@app.route('/classify', methods=['POST'])
def classify_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    image = request.files['image']
    image_pil = Image.open(image)
    
    # Predict the class and confidence
    predicted_class, confidence = predict_image(model, image_pil)

    # Save the uploaded image temporarily
    image_filename = secure_filename(image.filename)
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
    image_pil.save(image_path)

    # Construct the URL for the uploaded image
    image_url = url_for('uploaded_file', filename=image_filename)

    print("Image URL:", image_url)  # Debug statement

    # Render the result template with classification result and image URL
    return render_template('result.html', predicted_class=predicted_class, confidence=confidence, image_url=image_url)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
