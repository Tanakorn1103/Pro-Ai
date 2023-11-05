import cv2
import requests
import numpy as np
import pickle
import os
import base64

url = "http://localhost:8080/api/facevector"
def img2vec(img):
    v, buffer = cv2.imencode(".jpg", img)
    img_str = base64.b64encode(buffer)
    data = "image data,"+str.split(str(img_str),"'")[1]
    response = requests.get(url, json={"img":data})
    return response.json()

# กำหนด path ของโฟลเดอร์ที่มีรูปภาพ
path = 'train'

facvectors = []

# สร้าง category_mapping จากโฟลเดอร์ภายใน "train"
category_mapping = {}
for category in os.listdir(path):
    category_mapping[category] = len(category_mapping)

for sub in os.listdir(path):
    sub_folder_path = os.path.join(path, sub)

    # ตรวจสอบว่า sub_folder_path เป็นโฟลเดอร์
    if os.path.isdir(sub_folder_path):
        # หาค่าตำแหน่งของ sub ใน category_mapping
        category = category_mapping.get(sub, -1)
        if category != -1:
            for fn in os.listdir(sub_folder_path):
                img_file_name = os.path.join(sub_folder_path, fn)
                img = cv2.imread(img_file_name)
                # ทำการอ่านรูปภาพและนำไปแมปกับกลุ่ม (category)
                res = img2vec(img)
                vec = list(res["vector"])
                vec.append(category)
                facvectors.append(vec)

fcevectors = pickle.load(open('facevectors.pkl','rb'))
facevecors_np = np.array(fcevectors)
X_train = facevecors_np[:,0:16]
Y_train = facevecors_np[:,16]
print(X_train.shape)
print(Y_train.shape)