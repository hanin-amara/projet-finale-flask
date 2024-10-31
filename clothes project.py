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



# Image classification model
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    'static/tissu',
    target_size=(128, 128), 
    batch_size=32,
    class_mode='categorical' 
)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(train_generator.num_classes, activation='softmax')  
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_generator, epochs=10)

def predict_tissu(image_path):
    img = image.load_img(image_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    class_label = list(train_generator.class_indices.keys())
    
    return class_label[class_index]


@app.route('/cart')
def cart():
    image_path = 'static/images/tessu-chemise-p2.png'
    tissu_prediction = predict_tissu(image_path)
    return render_template('cart.html', tissu_prediction=tissu_prediction, image_path=image_path)

@app.route('/p2')
def p2():
    image_path = 'static/images/tisuu-pull.jpg'
    tissu_prediction = predict_tissu(image_path)
    return render_template('p2.html', tissu_prediction=tissu_prediction, image_path=image_path)

@app.route('/p3')
def p3():
    image_path = 'static/images/tissu-pant.png'
    tissu_prediction = predict_tissu(image_path)
    return render_template('p3.html', tissu_prediction=tissu_prediction, image_path=image_path)

@app.route('/p4')
def p4():  
    return render_template('p4.html')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9080)
