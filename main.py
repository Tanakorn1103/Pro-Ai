import numpy as np
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import base64
import pickle
import os

app = Flask(__name__)
CORS(app)


# สร้างฟังก์ชันเพื่อโหลดโมเดล CNN
def load_animal_cnn_model():
    model_filename = "animal_cnn_model.pkl"
    with open(model_filename, 'rb') as model_file:
        model = pickle.load(model_file)
    return model

# สร้างฟังก์ชันเพื่อปรับขนาดรูปภาพเป็น 128x128
def resize_image(image, size):
    return cv2.resize(image, (size, size))

# สร้างฟังก์ชันเพื่อพยากรณ์ประเภทของสัตว์จากรูปภาพ
def predict_animal_type(image, model):
    # ทำการปรับขนาดรูปภาพเป็น 128x128
    resized_image = resize_image(image, 128)
    
    # ทำการพยากรณ์ประเภทของสัตว์จากรูปภาพ
    prediction = model.predict(resized_image.reshape(1, 128, 128, 3))
    
    # ดึงคลาสที่มีความน่าจะเป็นสูงสุด
    predicted_class = np.argmax(prediction)
    path = 'train'
    
    # สร้าง mapping ระหว่างเลขหมายของคลาสกับชื่อของสัตว์
    class_mapping = {}
    for category in os.listdir(path):
        class_mapping[category] = len(class_mapping)
    
    #  ดึงชื่อของสัตว์จากเลขหมายของคลาส
    animal_type = list(class_mapping.keys())[list(class_mapping.values()).index(predicted_class)]
    
    return animal_type

@app.route('/api/')
def landing():
    return "This is a face to age API"

@app.route('/api/cnnmodel', methods=['POST'])
@cross_origin()
def get_animals_cnn():
    data = request.json.get('image_data')

    # Decode the base64 image data
    encoded_data = data
    decoded_data = base64.b64decode(encoded_data)

    # Load your CNN model
    model = load_animal_cnn_model()

    # Perform the image processing and prediction
    img = cv2.imdecode(np.frombuffer(decoded_data, np.uint8), cv2.IMREAD_COLOR)

    # ทำการปรับขนาดรูปภาพเป็น 128x128
    img = cv2.resize(img, (128, 128))

    animal_type = predict_animal_type(img, model)

    response_data = {"animal_type": animal_type}
    return jsonify(response_data)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
