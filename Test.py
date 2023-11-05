from tensorflow.keras.models import model_from_json
from autoencoder import ConvAutoencoder
import matplotlib.pyplot as plt
import numpy as np
import cv2
from flask import Flask, request
from flask_cors import CORS, cross_origin
import base64
import json

app = Flask(__name__)
cors = CORS(app)


json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
autoencoder = model_from_json(loaded_model_json)

autoencoder.load_weights("model.h5")



json_file = open('model_encoder.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
encoder = model_from_json(loaded_model_json)

encoder.load_weights("model_encoder.h5")



json_file = open('model_decoder.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
decoder = model_from_json(loaded_model_json)

decoder.load_weights("model_decoder.h5")

def readb64(uri):
   encoded_data = uri.split(',')[1]
   nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
   img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
   img = cv2.resize(img, (128, 128))
   return img

@app.route('/api/facevector', methods=['GET'])
@cross_origin()
def gen_feature():
    img_str = request.json['img']
    img = readb64(img_str)
    img_expandim = np.expand_dims(img, axis=0)
    encoded = np.array(encoder(img_expandim))
    print(encoded.shape)
    facevec = json.dumps({'vector':encoded[0].tolist()})
    return facevec
    

if __name__ == "__main__":
    app.run(host="0.0.0.0")