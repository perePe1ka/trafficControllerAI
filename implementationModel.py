import cv2
import logging
import numpy as np
from tensorflow.keras.models import load_model
from tkinter import filedialog

logging.getLogger('tensorflow').setLevel(logging.ERROR)
sizeA = 300
categories = ['neck', 'left', 'back', 'right', 'mouth', 'up']
model = load_model('modelv5.h5')


def predict_image(image):
    image = cv2.resize(image, (sizeA, sizeA))


image = np.expand_dims(image, axis=0)
image = np.expand_dims(image, axis=3)
image = image / 255.0
prediction = model.predict(image)
class_index = int(np.argmax(prediction))
confidence = float(prediction[0][class_index])
category = categories[class_index]
return category, confidence


def open_file():
    file_path = filedialog.askopenfilename()


if file_path:
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
category, confidence = predict_image(img)
if category == "neck" or category == "up" or category ==
    "back" or category == "right":
whatToDo = "Стоять"
elif category == "left":
whatToDo = "Можно ехать в любом направлении"
else:
whatToDo = "Можно ехать направо"
result_label.config(text=f"Жест:
{category}\nПредпологаемая
точность: {confidence}\nЧто
делать:
{whatToDo}
")
img = Image.open(file_path)
img = img.resize((300, 300), resample=Image.LANCZOS)
img = ImageTk.PhotoImage(img)
panel.config(image=img)
panel.image = img
root = tk.Tk()
root.title("Image Classifier")
root.geometry("800x800")
background_color = "#F0F0F0"
bg_label = tk.Label(root, bg=background_color)
bg_label.place(relwidth=1, relheight=1)
panel = tk.Label(root)
panel.pack(pady=10)
open_button = tk.Button(root, text="Открыть изображение",
                        command=open_file, bg="#FF5733", fg="white",
                        font=("Arial", 12))
open_button.pack(pady=10)
result_label = tk.Label(root, text="", font=("Arial", 12),
                        bg="#FF5733", fg="white")
result_label.pack(pady=10)
root.mainloop()
