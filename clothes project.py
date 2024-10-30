from flask import Flask, render_template, request, send_from_directory, jsonify
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.neighbors import NearestNeighbors
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['DATASET_FOLDER'] = 'static'
app.config['UPLOAD_FOLDER'] = 'uploads/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the pre-trained ResNet50 model for feature extraction
base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

# Function to extract image features using ResNet50
def extract_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224)) 
    img_data = image.img_to_array(img)  
    img_data = np.expand_dims(img_data, axis=0)  
    img_data = preprocess_input(img_data)  
    features = base_model.predict(img_data)  
    return features

# Load all images and extract their features from the dataset
def load_dataset_images(dataset_path):
    valid_extensions = ('.jpg', '.jpeg', '.png')
    features_list = []
    image_paths = []

    for img_file in os.listdir(dataset_path):
        img_path = os.path.join(dataset_path, img_file)
        if os.path.isfile(img_path) and img_file.lower().endswith(valid_extensions):
            features = extract_features(img_path)
            features_list.append(features)
            image_paths.append(img_file)

    features_list = np.array(features_list).reshape(len(features_list), -1) 
    return features_list, image_paths

# Load dataset images and their features
dataset_path = app.config['DATASET_FOLDER']
features_list, images = load_dataset_images(dataset_path)

# Train a KNN model to find similar images based on the extracted features
nbrs = NearestNeighbors(n_neighbors=7, metric='cosine').fit(features_list)

# Function to find similar images based on the uploaded image
def find_similar_images(uploaded_img_path):
    uploaded_features = extract_features(uploaded_img_path)
    distances, indices = nbrs.kneighbors(uploaded_features)

    similar_images = []
    distance_threshold = 0.5 
    for i, distance in enumerate(distances[0]):
        if distance < distance_threshold:
            similar_images.append(images[indices[0][i]])

    return similar_images

# Route to serve images from the dataset
@app.route('/static/<path:filename>')
def dataset_file(filename):
    return send_from_directory(app.config['DATASET_FOLDER'], filename)

# Route for file upload and similar image search
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        similar_images = find_similar_images(file_path)
        
        return jsonify(similar_images=similar_images) 

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)


@app.route('/cart')
def cart():
    return render_template('cart.html')

@app.route('/p2')
def p2():
    return render_template('p2.html')

@app.route('/p3')
def p3():
    return render_template('p3.html')

@app.route('/p4')
def p4():
    return render_template('p4.html')
@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9080)  
