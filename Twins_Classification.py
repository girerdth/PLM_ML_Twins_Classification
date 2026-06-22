"""
Created on Tues Feb 10 2026
@author: Thomas Girerd

This code generates the GUI for twins classification and grains segmentation.

"""
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import os
import copy
import numpy as np
from ultralytics import YOLO
import source_code.pseudoimage as pseudoimage
import math
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

        # Frame for image displays
        self.image_frame = tk.Frame(self.root)
        self.image_frame.pack(pady=10)

        # Labels for images
        self.image_label = tk.Label(self.image_frame)
        self.image_label.pack(side=tk.LEFT, padx=10)

        self.plm_label = tk.Label(self.image_frame)
        self.plm_label.pack(side=tk.LEFT, padx=10)

    def create_widgets(self):
        # Frame for buttons
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=10)

        # Buttons
        tk.Button(button_frame, text="Select Image", command=self.select_image).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Create Pseudocolour", command=self.create_pseudoimage).pack(side=tk.LEFT, padx=5)
        self.btn_simple  = tk.Button(button_frame, text="Simple Segmentation", command=self.run_simplify)
        self.btn_amplify = tk.Button(button_frame, text="Twins Classification", command=self.run_amplify)
        self.btn_simple.pack(side=tk.LEFT, padx=5)
        self.btn_amplify.pack(side=tk.LEFT, padx=5)
        self._set_processing_buttons(enabled=False)  # locked on startup

        # Resolution panel (hidden until image is loaded)
        self.res_frame = tk.Frame(self.root, bg="#f0f0f0", pady=6, padx=10)

        tk.Label(self.res_frame, text="Resolution:", bg="#f0f0f0").pack(side=tk.LEFT, padx=(0, 4))

        self.res_var = tk.StringVar()
        self.res_entry = tk.Entry(self.res_frame, textvariable=self.res_var, width=10)
        self.res_entry.pack(side=tk.LEFT)

        tk.Label(self.res_frame, text="µm / px", bg="#f0f0f0").pack(side=tk.LEFT, padx=(4, 12))

        tk.Button(self.res_frame, text="Confirm", command=self._confirm_resolution).pack(side=tk.LEFT)

        self.res_status = tk.Label(self.res_frame, text="⚠ Enter resolution to enable processing",
                                   fg="red", bg="#f0f0f0")
        self.res_status.pack(side=tk.LEFT, padx=(10, 0))

        # Resolution attribute (None until confirmed)
        self.resolution = None
        # Image display
        self.image_label = tk.Label(self.root)
        self.image_label.pack(pady=10)

    def select_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        if file_path:
            self.current_image = cv2.imread(file_path)
            self.current_image_name = os.path.basename(file_path)
            self.image_path = file_path
            self.display_image(self.current_image)
            self._show_resolution_panel()

    def _confirm_resolution(self):
        """Validate the entry and unlock processing if valid."""
        try:
            value = float(self.res_var.get())
            if value <= 0:
                raise ValueError
        except ValueError:
            self.res_status.config(text="✗ Enter a valid positive number", fg="red")
            self._set_processing_buttons(enabled=False)
            return
        self.resolution = value

        # Compute resolution after resizing to 640 px wide
        original_width = self.current_image.shape[1]
        scale_factor = original_width / 640
        self.resolution_640 = self.resolution * scale_factor

        THRESHOLD = 3.90e-1  # µm/px limit for the model

        if self.resolution_640 <= THRESHOLD:
            self.res_status.config(
                text=f"✓ {value:.4f} µm/px  |  {self.resolution_640:.4f} µm/px at 640 px  |  No cropping needed",
                fg="green"
            )
        else:
            self.res_status.config(
                text=(f"⚠ {value:.4f} µm/px  |  {self.resolution_640:.4f} µm/px at 640 px  |  "
                      f"Resolution out of training range — results will be unreliable"),
                fg="orange"
            )

        self._set_processing_buttons(enabled=True)

    def _show_resolution_panel(self):
        self.res_frame.pack(pady=4, before=self.image_frame)
        self.resolution = None
        self.resolution_640 = None
        self.n_crops = None
        self.res_var.set("")
        self.res_status.config(text="⚠ Enter resolution to enable processing", fg="red")
        self._set_processing_buttons(enabled=False)

    def create_pseudoimage(self):
        self.image_path, self.original_image = pseudoimage.main()
        print(self.image_path)
        if self.image_path is not None:
            self.display_image(cv2.imread(self.image_path))
            self._show_resolution_panel()
        else:
            messagebox.showerror("Error", "Failed to create pseudocolour image.")

    def run_simplify(self):
        if self.image_path is not None:
            result = simple(self.image_path)
            self.display_image_segmentation(result,self.image_path)
        else:
            messagebox.showerror("Error", "No image selected or created.")


    def _set_processing_buttons(self, enabled: bool):
        state = tk.NORMAL if enabled else tk.DISABLED
        self.btn_simple.config(state=state)
        self.btn_amplify.config(state=state)

    def run_amplify(self):
        if self.image_path is not None:
            result, plm_map = amplify(self.image_path, self.original_image)
            self.display_image_segment_plm_map(result,plm_map, self.image_path)
        else:
            messagebox.showerror("Error", "No image selected or created.")

    def display_image(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(image=img_pil)

        self.image_label.config(image=img_tk)
        self.image_label.image = img_tk  # Keep a reference

    def display_image_segmentation(self, img_contour, img_path):
        # 1. Load the pseudocolour image
        pseudocolour = cv2.imread(img_path)
        pseudocolour = cv2.cvtColor(pseudocolour, cv2.COLOR_BGR2RGB)

        # 2. Ensure the contour image is RGB
        # Assuming img_contour is the black/red/green/blue image you created earlier
        if len(img_contour.shape) == 2:  # If grayscale
            img_rgb = cv2.cvtColor(img_contour, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = cv2.cvtColor(img_contour, cv2.COLOR_BGR2RGB)

        # 3. Masking: We want to keep the CONTOURS (not white)
        # and show the PSEUDOCOLOUR image where the image is WHITE.
        white_mask = np.all(img_rgb == [255, 255, 255], axis=-1)

        # 4. Combine: Start with the contour image, then fill the background
        combined = img_rgb.copy()
        combined[white_mask] = pseudocolour[white_mask]

        # 5. Convert to Tkinter format
        img_pil = Image.fromarray(combined)
        img_tk = ImageTk.PhotoImage(image=img_pil)

        self.image_label.config(image=img_tk)
        self.image_label.image = img_tk


    def display_image_segment_plm_map(self, img_contour, plm_map, img_path):
        # 1. Load the pseudocolour background (the EBSD/orientation map)
        pseudocolour = cv2.imread(img_path)
        pseudocolour = cv2.cvtColor(pseudocolour, cv2.COLOR_BGR2RGB)

        # 2. Ensure the contour image is RGB
        if len(img_contour.shape) == 2:  # If grayscale
            img_rgb = cv2.cvtColor(img_contour, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = cv2.cvtColor(img_contour, cv2.COLOR_BGR2RGB)

        # 3. Masking: We want to keep the CONTOURS (not white)
        white_mask = np.all(img_rgb == [255, 255, 255], axis=-1)

        # 4. Combine: Start with the contour image, then fill the background
        combined = img_rgb.copy()
        combined[white_mask] = pseudocolour[white_mask]

        # 5. Convert to Tkinter format for contour image

        img_pil = Image.fromarray(combined)
        img_tk = ImageTk.PhotoImage(image=img_pil)

        # 6. Convert PLM map to Tkinter format
        plm_rgb = cv2.cvtColor(plm_map, cv2.COLOR_BGR2RGB)
        # 3. Masking: We want to keep the CONTOURS (not white)

        combined2 = img_rgb.copy()
        combined2[white_mask] = plm_rgb[white_mask]
        plm_pil = Image.fromarray(combined2)
        plm_tk = ImageTk.PhotoImage(image=plm_pil)

        # 7. Update the labels
        self.image_label.config(image=img_tk)
        self.image_label.image = img_tk  # Keep a reference

        self.plm_label.config(image=plm_tk)
        self.plm_label.image = plm_tk  # Keep a reference



if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()