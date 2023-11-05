import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import pickle

# กำหนดค่า hyperparameters
num_classes = 90  # จำนวนคลาสของสัตว์
input_shape = (128, 128, 3)  # ขนาดของรูปภาพ

# สร้างโมเดล CNN
model = Sequential()

# ชั้นการคอนโวลูชันและพูล
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

# ชั้นการแปรผล
model.add(Flatten())

# ชั้นเชื่อม
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

# ชั้นคลาสสิฟาย
model.add(Dense(num_classes, activation='softmax'))

# คอมไพล์โมเดลด้วย optimizer และค่าความสูญเสีย
model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# แสดงโครงสร้างของโมเดล
model.summary()

# บันทึกโมเดลลงในไฟล์
model_filename = "animal_cnn_model.pkl"
with open(model_filename, 'wb') as model_file:
    pickle.dump(model, model_file)