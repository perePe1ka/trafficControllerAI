Dropout
import cv2
import numpy as np
import os
from google.colab import drive

drive.mount('/content/drive')
sizeA = 300;
dataset_path = '/content/drive/MyDrive/obshiy/resizedDataSet'


def load_data(dataset_path):
    categories = ['neck', 'left', 'back', 'right', 'mouth', 'up']


data = []
labels = []
for category in categories:
    path = os.path.join(dataset_path, category)
class_num = categories.index(category)
for img in os.listdir(path):
    try:
        img_array = cv2.imread(os.path.join(path, img),
                               cv2.IMREAD_GRAYSCALE)
img_array = cv2.resize(img_array, (sizeA, sizeA))
data.append(img_array)
labels.append(class_num)
except Exception as e:
pass
data = np.array(data).reshape(-1, sizeA, sizeA, 1)
labels = np.array(labels)
return data, labels
data, labels = load_data(dataset_path)
gc.collect()
data = data / 255.0
gc.collect()
X_train, X_test, y_train, y_test = train_test_split(data, labels,
                                                    test_size=0.2, random_state=42)
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(sizeA, sizeA,
                                                             1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dense(6, activation='softmax'))
model.summary()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
datagen = ImageDataGenerator(
    rotation_range=0,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
)
checkpoint_path = '/content/drive/MyDrive/best_31.05.250.h5'
checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy',
                             verbose=1, save_best_only=True, mode='max')
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    validation_data=(X_test, y_test),
    steps_per_epoch=len(X_train) // 32,
    epochs=150,
    callbacks=[checkpoint]
)
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
model.save('/content/drive/MyDrive/modelv5.h5')
