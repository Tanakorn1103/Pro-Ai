from tensorflow.keras.optimizers import Adam
from autoencoder import ConvAutoencoder
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

# กำหนดค่า hyperparameters
EPOCHS = 100
BS = 32

# กำหนดเส้นทางที่เก็บรูปภาพ
path = 'D:\\AI\\archive\\animals\\animals\\'

# กำหนดรายการเก็บข้อมูลสำหรับฝึกและทดสอบ
train_x = []
test_x = []

# จำนวนรูปภาพที่คุณต้องการใช้ในการฝึกและทดสอบ
no = 5000
i = 0

# วนลูปผ่านโฟลเดอร์และไฟล์รูปภาพในเส้นทาง
for folder in os.listdir(path):
    for filename in os.listdir(os.path.join(path, folder)):
        img = cv2.imread(path + folder + '//' + filename, 1)
        
        # ใช้ cv2.resize เพื่อปรับขนาดรูปภาพให้มีขนาดเดียวกัน
        img = cv2.resize(img, (128, 128))  # ปรับขนาดเป็น 128x128
        
        if i < 4500:
            train_x.append(img)
        else:
            test_x.append(img)
        i += 1
        if (i == no):
            break
    if (i == no):
        break

# แปลงข้อมูลเป็น NumPy arrays
trainX = np.array(train_x)
testX = np.array(test_x)

# เพิ่มมิติเพื่อให้สามารถนำเข้าไปในโมเดล ConvAutoencoder
trainX = np.expand_dims(trainX, axis=-1)
testX = np.expand_dims(testX, axis=-1)

# ทำการปรับค่าความเข้มสีให้อยู่ในช่วง 0-1
trainX = trainX.astype("float32") / 255.0
testX = testX.astype("float32") / 255.0

# สร้าง Convolutional Autoencoder
(encoder, decoder, autoencoder) = ConvAutoencoder.build(128, 128, 3)
opt = Adam(lr=1e-3)
autoencoder.compile(loss="mse", optimizer=opt)

# ฝึก Convolutional Autoencoder
H = autoencoder.fit(
    trainX, trainX,
    validation_data=(testX, testX),
    epochs=EPOCHS,
    batch_size=BS)

# บันทึกโมเดล Autoencoder และส่วนประกอบต่าง ๆ เป็นไฟล์ JSON และ HDF5

# บันทึกโมเดล Autoencoder
model_json = autoencoder.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
autoencoder.save_weights("model.h5")
print("Saved model to disk")

# บันทึกโมเดล Encoder
model_encoder_json = encoder.to_json()
with open("model_encoder.json", "w") as json_file:
    json_file.write(model_encoder_json)
encoder.save_weights("model_encoder.h5")
print("Saved encoder model to disk")

# บันทึกโมเดล Decoder
model_decoder_json = decoder.to_json()
with open("model_decoder.json", "w") as json_file:
    json_file.write(model_decoder_json)
decoder.save_weights("model_decoder.h5")
print("Saved decoder model to disk")

# สร้างกราฟแสดงค่า Loss ของการฝึก
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig('loss.png')
