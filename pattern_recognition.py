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
        tk.Frame.__init__(self, master, pady=0)  # relief=tk.SUNKEN, bd=2)

        self.master.title("Pattern recognition")
        self.master.resizable(False, False)
        self.placeholder_filename = "no-image.jpg"
        self.filename = None
        self.inital_dir = os.getcwd()
        self.cv2_img = None
        self.img = None
        self.check_buttons = []
        self.checked_values = [IntVar() for i in range(20)]
        self.save_image_button = None
        self.modified_pil_img = None
        self.status_var = StringVar()
        self.curr_img_var = StringVar()

        self.setup_image_frame()
        self.setup_cb_frame()
        self.setup_load_save_frame()
        self.setup_menu()
        self.setup_status_bar()

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
        self.master.grid_rowconfigure(3, weight=1)


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
        self.apply_all_button = Button(cb_frame, text="Apply all", command=self.apply_all_patterns)

        for t in self.check_buttons:
            t[0].pack(anchor=W, fill=None, expand=False)

        self.apply_button.pack(pady=10, fill=None, expand=False, side=LEFT)
        self.clear_button.pack(padx=10, pady=10, fill=None, expand=False, side=RIGHT)
        self.apply_all_button.pack(padx=5, pady=10, fill=None, expand=None)

        cb_frame.grid(row=0, column=1, sticky=W)

    def setup_load_save_frame(self):
        load_save_frame = Frame(self, pady=10)
        load_image_button = Button(load_save_frame, text="Load image", command=self.load_image)
        self.save_image_button = Button(load_save_frame, text="Save modified image", command=self.save_image, state=DISABLED)
        load_image_button.pack(side=LEFT)
        self.save_image_button.pack(side=LEFT, padx=10)
        load_save_frame.grid(row=2, column=0)

    def setup_menu(self):
        menu = Menu(self.master)
        self.master.config(menu=menu)

        file_submenu = Menu(menu)
        menu.add_cascade(label="File", menu=file_submenu)
        file_submenu.add_command(label="New image", command=self.load_image)
        file_submenu.add_separator()
        file_submenu.add_command(label="Save changes", command=self.save_changes)
        file_submenu.add_command(label="Save as...", command=self.save_image)
        file_submenu.add_separator()
        file_submenu.add_command(label="Exit", command=self.quit)

        edit_submenu = Menu(menu)
        menu.add_cascade(label="Edit", menu=edit_submenu)
        edit_submenu.add_command(label="Apply selected patterns", command=self.apply_pattern_recognition)
        edit_submenu.add_command(label="Apply all patterns", command=self.apply_all_patterns)
        edit_submenu.add_separator()
        edit_submenu.add_command(label="Restore original image", command=self.restore_original_image)

        help_submenu = Menu(menu)
        menu.add_cascade(label="Help", menu=help_submenu)
        help_submenu.add_command(label="About", command=self.display_about_info)

    def setup_status_bar(self):
        status_bar = Frame(self, bd=1, relief=SUNKEN)
        self.status_var.set("Status: None")
        self.curr_img_var.set("Image: No image placeholder")
        status = Label(status_bar, anchor=W, textvariable=self.status_var)
        curr_img = Label(status_bar, anchor=W, textvariable=self.curr_img_var)
        curr_img.pack(side=LEFT, fill=BOTH, expand=True)
        status.pack(side=LEFT, fill=BOTH, expand=True)
        status_bar.grid(row=3, column=0, columnspan=2, sticky="ew")

    def set_status(self, status="", image=""):
        self.status_var.set(status)
        self.curr_img_var.set(image)

    def clear_status(self):
        self.set_status("")

    def load_image(self, filename=None):
        try:
            if filename:
                self.filename = filename
            else:
                self.reset()
                self.filename = filedialog.askopenfilename(initialdir=self.inital_dir, title="Select image",
                                                           filetypes=(("jpeg files", "*.jpg"), ("png files", "*.png")))
                self.inital_dir = os.path.dirname(self.filename)
            if self.filename:
                self.cv2_img = cv2.imread(self.filename)
                b, g, r = cv2.split(self.cv2_img)
                self.img = cv2.merge((r, g, b))

                self.original_pil_img = Image.fromarray(self.img)
                self.original_pil_img.thumbnail((600, 600))
                self.tk_photo = ImageTk.PhotoImage(self.original_pil_img)
                self.image_label.configure(image=self.tk_photo)
            self.set_status(status="Status: Successfully loaded image.", image=str(self.filename))
        except Exception as e:
            messagebox.showerror("Error", "An error occurred while loading image: \n" + str(e))
            self.set_status(status="Error occured", image=str(self.filename))

    def save_image(self, file=None):
        try:
            if not file:
                file = filedialog.asksaveasfile(mode='w', defaultextension=".jpg")
            if file:
                if self.modified_pil_img:
                    self.modified_pil_img.save(file)
                else:
                    self.original_pil_img.save(file)
            self.set_status(status="Image saved successfully.", image=str(self.filename))
        except Exception as e:
            messagebox.showerror("Error", "An error occurred while saving image: \n" + str(e))

    def save_changes(self):
        if self.modified_pil_img:
            try:
                self.modified_pil_img.save(self.filename)
                messagebox.showinfo("Success", "All changes saved successfully.")
            except Exception as e:
                messagebox.showerror("Error", "An error occurred while saving changes: \n" + str(e))
            self.set_status(status="Changes saved successfully.", image=str(self.filename))
        else:
            messagebox.showinfo("Nothing to save", "Original image has not been modified.")

    def apply_pattern_recognition(self, apply_all=False):
        any_selected = False
        allowed_labels = []
        for t in self.check_buttons:
            if self.checked_values[t[1]].get() or apply_all:
                self.save_image_button.config(state=NORMAL)
                any_selected = True
                allowed_labels.append(t[1])
        if any_selected:
            self.set_status(status="Status: Processing...", image=str(self.filename))
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
                messagebox.showinfo("Some objects not recognized",
                    "The following objects are not recognized: \n" + label_string)
            self.set_status(status="Status: Pattern recognition done.", image=str(self.filename))
        else:
            self.tk_photo = ImageTk.PhotoImage(self.original_pil_img)
            self.image_label.configure(image=self.tk_photo)
            self.save_image_button.config(state=DISABLED)
            messagebox.showinfo("Info", "Select one or more shapes/patterns.")

    def apply_all_patterns(self):
        self.apply_pattern_recognition(True)

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

    def restore_original_image(self):
        self.load_image(self.filename)
        self.tk_photo = ImageTk.PhotoImage(self.original_pil_img)
        self.image_label.configure(image=self.tk_photo)
        self.set_status(status="Original image restored.", image=str(self.filename))

    def display_about_info(self):
        messagebox.showinfo(title="About pattern recognition app",
                            message="======Insert info======")

    def quit(self):
        self.master.destroy()


def pattern_recognition_app():
    root = Tk()
    app = PatternRecognitionApp(root)
    root.mainloop()


pattern_recognition_app()