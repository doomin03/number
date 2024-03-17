import tkinter as tk
from tkinter import Canvas, Button, messagebox
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image, ImageDraw

# 숫자를 그릴 캔버스 크기
CANVAS_SIZE = 280

class NumberRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("숫자 맞추기 프로그램")

        self.canvas = Canvas(root, width=CANVAS_SIZE, height=CANVAS_SIZE, bg="white")
        self.canvas.pack()

        self.reset_button = Button(root, text="리셋", command=self.reset_canvas)
        self.reset_button.pack()

        self.recognize_button = Button(root, text="숫자 인식", command=self.recognize_number)
        self.recognize_button.pack()

        self.model = load_model("model")  # 학습된 모델 로드

        self.canvas.bind("<B1-Motion>", self.draw)

        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE))
        self.draw = ImageDraw.Draw(self.image)

    def draw(self, event):
        x1, y1 = (event.x - 5), (event.y - 5)
        x2, y2 = (event.x + 5), (event.y + 5)
        self.canvas.create_oval(x1, y1, x2, y2, fill="black")
        self.draw.ellipse([x1, y1, x2, y2], fill="white")

    def reset_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE))
        self.draw = ImageDraw.Draw(self.image)

    def recognize_number(self):
        resized_image = self.image.resize((28, 28))
        grayscale_image = resized_image.convert("L")
        image_array = np.array(grayscale_image)
        image_array = image_array.reshape(1, 28, 28, 1)
        image_array = image_array / 255.0

        prediction = self.model.predict(image_array)
        predicted_number = np.argmax(prediction)
        messagebox.showinfo("숫자 예측 결과", f"모델이 예측한 숫자: {predicted_number}")

if __name__ == "__main__":
    root = tk.Tk()
    app = NumberRecognizerApp(root)
    root.mainloop()
