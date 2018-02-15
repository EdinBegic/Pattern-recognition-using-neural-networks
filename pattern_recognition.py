import os
import tkinter as tk
import cv2
from tkinter import filedialog, messagebox
from tkinter import *
from PIL import Image, ImageTk
from object_recognition import Object_Detector

PROTOTXT_PATH = "MobileNetSSD_deploy.prototxt.txt"
MODEL_PATH = "MobileNetSSD_deploy.caffemodel"
class PatternRecognitionApp(tk.Frame):
    def __init__(self, master=None):
        tk.Frame.__init__(self, master, relief=tk.SUNKEN, bd=2)

        self.master.title("Pattern recognition")
        self.master.resizable(False, False)
        self.placeholder_filename = "no-image.jpg"
        self.filename = None
        self.cv2_img = None
        self.img = None
        self.check_buttons = []
        self.checked_values = [IntVar() for i in range(20)]
        self.save_image_button = None

        self.setup_image_frame()
        self.setup_cb_frame()
        self.setup_load_save_frame()

        exit_button = Button(self, text="Close", command=self.quit)
        exit_button.grid(row=2, column=1)
        self.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=10)
        self.configure_weights()

        separator = Frame(self, height=5, bd=1, relief=SUNKEN)
        separator.grid(row=1, columnspan=2, padx=5, pady=5, sticky="ew")

    def configure_weights(self):
        self.master.grid_columnconfigure(0, weight=1)
        self.master.grid_columnconfigure(1, weight=1)
        self.master.grid_rowconfigure(0, weight=1)
        self.master.grid_rowconfigure(1, weight=1)
        self.master.grid_rowconfigure(2, weight=1)

    def setup_image_frame(self):
        image_frame = Frame(self, padx=20, pady=20)
        self.image_label = Label(image_frame, padx=20, pady=20, bg="grey")
        self.load_image(self.placeholder_filename)
        self.image_label.pack()
        image_frame.grid(row=0, column=0)

    def setup_cb_frame(self):
        cb_frame = Frame(self, padx=50)
        choose_pattern_label = Label(cb_frame, pady=5, text="Choose pattern for recognition:")
        choose_pattern_label.pack(fill=None, expand=False)

        self.check_buttons.append((Checkbutton(cb_frame, text="Airplane", variable=self.checked_values[1]), 1))
        self.check_buttons.append((Checkbutton(cb_frame, text="Bicycle", variable=self.checked_values[2]), 2))
        self.check_buttons.append((Checkbutton(cb_frame, text="Bird", variable=self.checked_values[3]), 3))
        self.check_buttons.append((Checkbutton(cb_frame, text="Boat", variable=self.checked_values[4]), 4))
        self.check_buttons.append((Checkbutton(cb_frame, text="Bottle", variable=self.checked_values[5]), 5))
        self.check_buttons.append((Checkbutton(cb_frame, text="Bus", variable=self.checked_values[6]), 6))
        self.check_buttons.append((Checkbutton(cb_frame, text="Car", variable=self.checked_values[7]), 7))
        self.check_buttons.append((Checkbutton(cb_frame, text="Cow", variable=self.checked_values[10]), 10))
        self.check_buttons.append((Checkbutton(cb_frame, text="Dog", variable=self.checked_values[12]), 12))
        self.check_buttons.append((Checkbutton(cb_frame, text="Horse", variable=self.checked_values[13]), 13))
        self.check_buttons.append((Checkbutton(cb_frame, text="Person", variable=self.checked_values[15]), 15))

        self.apply_button = Button(cb_frame, text="Apply", command=self.apply_pattern_recognition)
        self.clear_button = Button(cb_frame, text="Clear all", command=self.clear_all)

        for t in self.check_buttons:
            t[0].pack(anchor=W, fill=None, expand=False)

        self.apply_button.pack(pady=10, fill=None, expand=False, side=LEFT)
        self.clear_button.pack(padx=10, pady=10, fill=None, expand=False, side=RIGHT)
        cb_frame.grid(row=0, column=1, sticky=W)

    def setup_load_save_frame(self):
        load_save_frame = Frame(self)
        load_image_button = Button(load_save_frame, text="Load image", command=self.load_image)
        self.save_image_button = Button(load_save_frame, text="Save modified image", command=self.save_image, state=DISABLED)
        load_image_button.pack(side=LEFT)
        self.save_image_button.pack(side=LEFT, padx=10)
        load_save_frame.grid(row=2, column=0)

    def load_image(self, filename=None):
        if filename:
            self.filename = filename
        else:
            self.reset()
            self.filename = filedialog.askopenfilename(initialdir=os.getcwd(), title="Select image",
                                                       filetypes=(("jpeg files", "*.jpg"), ("png files", "*.png")))
        if self.filename:
            self.cv2_img = cv2.imread(self.filename)
            b, g, r = cv2.split(self.cv2_img)
            self.img = cv2.merge((r, g, b))

            self.original_pil_img = Image.fromarray(self.img)
            self.original_pil_img.thumbnail((600, 600))
            self.tk_photo = ImageTk.PhotoImage(self.original_pil_img)
            self.image_label.configure(image=self.tk_photo)

    def save_image(self):
        file = filedialog.asksaveasfile(mode='w', defaultextension=".jpg")
        if file:
            self.modified_pil_img.save(file)

    def apply_pattern_recognition(self):
        any_selected = False
        allowed_labels = []
        for t in self.check_buttons:
            if self.checked_values[t[1]].get():
                self.save_image_button.config(state=NORMAL)
                any_selected = True
                # napomena na srbskom:
                # ovo 'num' je redni broj iz one tvoje liste
                # self.cv2_img sadrzi trenutnu sliku u cv2 formatu
                allowed_labels.append(t[1])
                # INJECT BEGA'S RECOGNITION LOGIC
                # UPDATE VIEW
                # self.update_image(modified_cv2_img) # slika koja je rezultat procesiranja ovog detektora
        if any_selected:
            object_detector = Object_Detector()
            object_detector.set_prototxt_path(PROTOTXT_PATH)
            object_detector.set_model_path(MODEL_PATH)
            object_detector.set_image(self.cv2_img)
            object_detector.load_model()
            object_detector.load_image()
            object_detector.forward_propagation_blob()
            not_found_labels = object_detector.object_labeling(allowed_labels)
            self.update_image(object_detector.get_image())
            if len(not_found_labels) > 0:
                label_string = '\n'.join(str(l) for l in not_found_labels)
                messagebox.showwarning("Some objects not recognized", 
                    "The following objects are not recognized: \n" + label_string)
        if not any_selected:
            self.tk_photo = ImageTk.PhotoImage(self.original_pil_img)
            self.image_label.configure(image=self.tk_photo)
            self.save_image_button.config(state=DISABLED)

    def update_image(self, cv2_img):
        b, g, r = cv2.split(cv2_img)
        modified_img = cv2.merge((r, g, b))

        self.modified_pil_img = Image.fromarray(modified_img)
        self.modified_pil_img.thumbnail((600, 600))
        self.tk_photo = ImageTk.PhotoImage(self.modified_pil_img)
        self.image_label.configure(image=self.tk_photo)

    def clear_all(self):
        for t in self.check_buttons:
            t[0].deselect()

    def reset(self):
        self.clear_all()
        self.save_image_button.config(state=DISABLED)
        self.modified_pil_img = None

    def quit(self):
        self.master.destroy()


def pattern_recognition_app():
    root = Tk()
    app = PatternRecognitionApp(root)
    root.mainloop()


pattern_recognition_app()