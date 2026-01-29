import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import os
import copy
import numpy as np
from ultralytics import YOLO
import source_code.pseudoimage as pseudoimage
from source_code.run_models import simplify_method as simple
from source_code.run_models import amplify_method as amplify

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Pseudocolour Image Processor")

        # Variables
        self.current_image = None
        self.current_image_name = None
        self.displayed_image = None

        # GUI Elements
        self.create_widgets()

    def create_widgets(self):
        # Frame for buttons
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=10)

        # Buttons
        tk.Button(button_frame, text="Select Image", command=self.select_image).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Create Pseudocolour", command=self.create_pseudoimage).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Simple Segmentation", command=self.run_simplify).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Advanced Segmentation", command=self.run_amplify).pack(side=tk.LEFT, padx=5)

        # Image display
        self.image_label = tk.Label(self.root)
        self.image_label.pack(pady=10)

    def select_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        if file_path:
            self.current_image = cv2.imread(file_path)
            self.current_image_name = os.path.basename(file_path)
            self.display_image(self.current_image)

    def create_pseudoimage(self):
        self.image_path, self.original_image = pseudoimage.main()
        print(self.image_path)
        if self.image_path is not None:
            self.display_image(cv2.imread(self.image_path))
        else:
            messagebox.showerror("Error", "Failed to create pseudocolour image.")

    def run_simplify(self):
        if self.image_path is not None:
            result = simple(self.image_path)
            self.display_image_segmentation(result,self.image_path)
        else:
            messagebox.showerror("Error", "No image selected or created.")

    def run_amplify(self):
        if self.image_path is not None:
            result = amplify(self.image_path, self.original_image)
            self.display_image_segmentation(result,self.image_path)
        else:
            messagebox.showerror("Error", "No image selected or created.")

    def display_image(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(image=img_pil)

        self.image_label.config(image=img_tk)
        self.image_label.image = img_tk  # Keep a reference

    def display_image_segmentation(self, img, img_path):
        pseudocolour = cv2.imread(img_path)
        pseudocolour = cv2.cvtColor(pseudocolour, cv2.COLOR_BGR2RGB)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Create mask for white pixels (all channels == 255)
        white_mask = np.all(img_rgb == 255, axis=2)

        # Initialize output
        toto = img_rgb.copy()

        # Replace pixels where img_rgb is white
        toto[white_mask] = pseudocolour[white_mask]

        img_pil = Image.fromarray(toto)
        img_tk = ImageTk.PhotoImage(image=img_pil)

        self.image_label.config(image=img_tk)
        self.image_label.image = img_tk  # Keep a reference

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()