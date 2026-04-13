import os
from flask import Flask, render_template, request, redirect, send_from_directory, url_for
from werkzeug.utils import secure_filename
import numpy as np
import json
import uuid
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
UPLOAD_FOLDER = os.path.join(ROOT_DIR, "uploadimages")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(
    __name__,
    template_folder=os.path.join(ROOT_DIR, "template"),
    static_folder=os.path.join(ROOT_DIR, "static"),
    static_url_path="/static"
)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
model = tf.keras.models.load_model(os.path.join(ROOT_DIR, "models", "plant_disease_recog_model_pwp.keras"))
label = ['Apple___Apple_scab',
 'Apple___Black_rot',
 'Apple___Cedar_apple_rust',
 'Apple___healthy',
 'Background_without_leaves',
 'Blueberry___healthy',
 'Cherry___Powdery_mildew',
 'Cherry___healthy',
 'Corn___Cercospora_leaf_spot Gray_leaf_spot',
 'Corn___Common_rust',
 'Corn___Northern_Leaf_Blight',
 'Corn___healthy',
 'Grape___Black_rot',
 'Grape___Esca_(Black_Measles)',
 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
 'Grape___healthy',
 'Orange___Haunglongbing_(Citrus_greening)',
 'Peach___Bacterial_spot',
 'Peach___healthy',
 'Pepper,_bell___Bacterial_spot',
 'Pepper,_bell___healthy',
 'Potato___Early_blight',
 'Potato___Late_blight',
 'Potato___healthy',
 'Raspberry___healthy',
 'Soybean___healthy',
 'Squash___Powdery_mildew',
 'Strawberry___Leaf_scorch',
 'Strawberry___healthy',
 'Tomato___Bacterial_spot',
 'Tomato___Early_blight',
 'Tomato___Late_blight',
 'Tomato___Leaf_Mold',
 'Tomato___Septoria_leaf_spot',
 'Tomato___Spider_mites Two-spotted_spider_mite',
 'Tomato___Target_Spot',
 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 'Tomato___Tomato_mosaic_virus',
 'Tomato___healthy']

with open(os.path.join(BASE_DIR, "plant_disease.json"),'r') as file:
    plant_disease = json.load(file)

# print(plant_disease[4])

@app.route('/uploadimages/<path:filename>')
def uploaded_images(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/',methods = ['GET'])
def home():
    return render_template('home.html')

def extract_features(image_path):
    image_data = tf.io.read_file(image_path)
    image = tf.image.decode_image(image_data, channels=3, expand_animations=False)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [160, 160])
    image = tf.expand_dims(image, axis=0)
    return image

def model_predict(image):
    img = extract_features(image)
    prediction = model.predict(img)
    # print(prediction)
    prediction_label = plant_disease[prediction.argmax()]
    return prediction_label

@app.route('/upload/',methods = ['POST','GET'])
def uploadimage():
    if request.method == "POST":
        image = request.files['img']
        safe_filename = secure_filename(image.filename)
        if not safe_filename:
            safe_filename = f"image_{uuid.uuid4().hex}.jpg"
        image_filename = f"temp_{uuid.uuid4().hex}_{safe_filename}"
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
        image.save(image_path)
        print(image_path)
        prediction = model_predict(image_path)
        image_url = url_for('uploaded_images', filename=image_filename)
        return render_template('home.html', result=True, imagepath=image_url, prediction=prediction)
    else:
        return redirect('/')
        
    
if __name__ == "__main__":
    app.run(debug=True)