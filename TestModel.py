import tensorflow as tf
from tensorflow.keras.models import Sequential

Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import cv2
from google.colab import drive

drive.mount('/content/drive')
sizeA = 254;
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

dataset_path = '/content/drive/MyDrive/testDataset'
categories = ['neck', 'left', 'back', 'right', 'mouth', 'up']


def load_data(dataset_path):
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
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data, labels,
                                                    test_size=0.2, random_state=42)
model_path = '/content/drive/MyDrive/modelv5.h5'
model = load_model(model_path)
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print(f"Test Loss: {loss:.4f}")
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
conf_matrix = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
class_names = ['neck', 'left', 'back', 'right', 'mouth', 'up']
report = classification_report(y_test, y_pred_classes,
                               target_names=class_names)
print(report)
