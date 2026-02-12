import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from src.predict import Predictor
import os


class PredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Pet Classifier - Cat vs Dog")
        self.root.geometry("500x600")
        self.root.configure(bg="#f0f0f0")

        # Initialize the Predictor from your src folder
        try:
            self.predictor = Predictor()
        except Exception as e:
            messagebox.showerror("Error", f"Could not load model: {e}")
            self.root.destroy()

        # UI Elements
        self.label = tk.Label(root, text="Cat vs Dog Classifier", font=("Helvetica", 20, "bold"), bg="#f0f0f0")
        self.label.pack(pady=20)

        self.canvas = tk.Canvas(root, width=300, height=300, bg="white", highlightthickness=1)
        self.canvas.pack(pady=10)

        self.btn_browse = tk.Button(root, text="Select Image", command=self.load_image,
                                    font=("Helvetica", 12), bg="#4CAF50", fg="white", padx=20)
        self.btn_browse.pack(pady=20)

        self.result_label = tk.Label(root, text="Result: None", font=("Helvetica", 16), bg="#f0f0f0")
        self.result_label.pack(pady=10)

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if not file_path:
            return

        # Show the image in the UI
        img = Image.open(file_path)
        img.thumbnail((300, 300))
        self.img_tk = ImageTk.PhotoImage(img)
        self.canvas.create_image(150, 150, image=self.img_tk)

        # Run Prediction
        label, confidence = self.predictor.predict(file_path)

        # Update UI
        color = "#2196F3" if label == "DOG" else "#FF5722"
        self.result_label.config(text=f"Result: {label} ({confidence:.2%})", fg=color)


if __name__ == "__main__":
    root = tk.Tk()
    app = PredictionApp(root)
    root.mainloop()