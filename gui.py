#!/usr/bin/env python3
"""
GUI for Face Recognition Attendance System
Tabbed interface for adding people, managing groups, and running face recognition.
Groups are saved to groups.json. Attendance is saved as Excel in Predict tab.
Includes console output for debugging in Add Person and Predict tabs.
Output video and Excel files named as <YYYYMMDD_HHMM>_R<random>_attendance.xlsx and <YYYYMMDD_HHMM>_R<random>.mp4.
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
from pathlib import Path
import json
import logging
from datetime import datetime
import cv2
import numpy as np
from ultralytics import YOLO
from face_processing_pipeline import process_single_image, process_video, process_person_folder, generate_embeddings
from predict import load_face_database, recognize_face_ensemble, draw_enhanced_bbox, predict_video
import shutil
import pandas as pd
import random

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize face detector
face_detector = YOLO("yolov8x-face.pt")

class TextHandler(logging.Handler):
    """Custom logging handler to output logs to a Tkinter Text widget."""
    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget
        self.text_widget.config(state='disabled')

    def emit(self, record):
        msg = self.format(record)
        self.text_widget.config(state='normal')
        self.text_widget.insert(tk.END, msg + '\n')
        self.text_widget.see(tk.END)
        self.text_widget.config(state='disabled')

class AttendanceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition Attendance System")
        self.root.geometry("1000x800")
        self.root.resizable(True, True)
        self.root.configure(bg='#1E1E1E')

        # Initialize groups and database
        self.groups = {"All": []}
        self.face_database = {}
        self.last_output_folder = None
        self.group_combobox = None
        self.load_groups()
        self.populate_people_list()

        # Set VSCode-like theme
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TButton', padding=8, font=('Helvetica', 11), background='#007ACC', foreground='#CCCCCC', borderwidth=0)
        style.map('TButton', background=[('active', '#005F9E')])
        style.configure('TLabel', font=('Helvetica', 11), foreground='#CCCCCC', background='#1E1E1E')
        style.configure('TEntry', padding=6, font=('Helvetica', 10), fieldbackground='#252526', foreground='#CCCCCC')
        style.configure('TCombobox', padding=6, font=('Helvetica', 10), fieldbackground='#252526', foreground='#CCCCCC')
        style.configure('TFrame', background='#1E1E1E')
        style.configure('TLabelframe', background='#1E1E1E', foreground='#CCCCCC')
        style.configure('TLabelframe.Label', background='#1E1E1E', foreground='#CCCCCC', font=('Helvetica', 12))
        style.configure('TNotebook', background='#1E1E1E', tabmargins=5)
        style.configure('TNotebook.Tab', background='#252526', foreground='#CCCCCC', font=('Helvetica', 11))
        style.map('TNotebook.Tab', background=[('selected', '#007ACC')])
        style.configure('TListbox', background='#252526', foreground='#CCCCCC', selectbackground='#264F78')

        # Create main tabbed interface
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=20, pady=20)

        # Create tabs
        self.add_person_tab = ttk.Frame(self.notebook)
        self.groups_tab = ttk.Frame(self.notebook)
        self.predict_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.add_person_tab, text="Add Person")
        self.notebook.add(self.groups_tab, text="Groups")
        self.notebook.add(self.predict_tab, text="Predict")

        # Initialize tab contents
        self.create_add_person_tab()
        self.create_groups_tab()
        self.create_predict_tab()

    def load_groups(self):
        """Load groups from groups.json."""
        groups_file = Path("groups.json")
        if groups_file.exists():
            try:
                with open(groups_file, "r") as f:
                    data = json.load(f)
                    self.groups = data.get("groups", {"All": []})
                    self.last_output_folder = data.get("last_output_folder")
                    logger.debug(f"Loaded groups: {self.groups}")
                    logger.debug(f"Last output folder: {self.last_output_folder}")
            except Exception as e:
                logger.error(f"Failed to load groups.json: {e}")
                messagebox.showerror("Error", f"Failed to load groups: {e}")

    def save_groups(self):
        """Save groups to groups.json."""
        groups_file = Path("groups.json")
        data = {
            "groups": self.groups,
            "last_output_folder": self.last_output_folder or str(Path("output_videos").absolute())
        }
        try:
            with open(groups_file, "w") as f:
                json.dump(data, f, indent=4)
            logger.debug("Saved groups to groups.json")
        except Exception as e:
            logger.error(f"Failed to save groups.json: {e}")
            messagebox.showerror("Error", f"Failed to save groups: {e}")

    def populate_people_list(self):
        """Populate the list of people from face_images directory."""
        face_images_dir = Path("face_images")
        if face_images_dir.exists():
            self.groups["All"] = sorted([d.name for d in face_images_dir.iterdir() if d.is_dir()])
        self.face_database = load_face_database(people=self.groups["All"])
        logger.debug(f"Populated people list: {self.groups['All']}")
        logger.debug(f"Face database keys: {list(self.face_database.keys())}")
        self.save_groups()

    def create_add_person_tab(self):
        """Create the Add Person tab with console."""
        frame = ttk.Frame(self.add_person_tab, padding="20")
        frame.pack(fill='both', expand=True)

        # Input Frame
        input_frame = ttk.LabelFrame(frame, text="Add New Person", padding="15", relief="groove")
        input_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N), pady=10)
        input_frame.columnconfigure(1, weight=1)

        # Person Name
        ttk.Label(input_frame, text="Person Name:").grid(row=0, column=0, sticky=tk.W, pady=8)
        self.person_name_entry = ttk.Entry(input_frame)
        self.person_name_entry.grid(row=0, column=1, columnspan=3, padx=5, pady=8, sticky=(tk.W, tk.E))

        # Media Source Selection
        ttk.Label(input_frame, text="Media Source:").grid(row=1, column=0, sticky=tk.W, pady=8)
        self.media_source_var = tk.StringVar(value="image")
        ttk.Radiobutton(input_frame, text="Image", variable=self.media_source_var, value="image").grid(row=1, column=1, sticky=tk.W, padx=5)
        ttk.Radiobutton(input_frame, text="Video", variable=self.media_source_var, value="video").grid(row=1, column=2, sticky=tk.W, padx=5)
        ttk.Radiobutton(input_frame, text="Folder", variable=self.media_source_var, value="folder").grid(row=1, column=3, sticky=tk.W, padx=5)

        # File/Folder Selection
        ttk.Label(input_frame, text="Select File/Folder:").grid(row=2, column=0, sticky=tk.W, pady=8)
        self.file_entry = ttk.Entry(input_frame)
        self.file_entry.grid(row=2, column=1, columnspan=2, padx=5, pady=8, sticky=(tk.W, tk.E))
        ttk.Button(input_frame, text="Browse", command=self.browse_file).grid(row=2, column=3, padx=10, pady=8)

        # Quality Threshold
        ttk.Label(input_frame, text="Quality Threshold:").grid(row=3, column=0, sticky=tk.W, pady=8)
        self.quality_var = tk.StringVar(value="35")
        ttk.Entry(input_frame, textvariable=self.quality_var, width=10).grid(row=3, column=1, sticky=tk.W, padx=5, pady=8)

        # Add Person Button
        ttk.Button(input_frame, text="Add Person", command=self.add_person).grid(row=4, column=0, columnspan=4, pady=15, sticky=tk.EW)

        # Console Output
        console_frame = ttk.LabelFrame(frame, text="Console Output", padding="10", relief="groove")
        console_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        console_frame.columnconfigure(0, weight=1)
        console_frame.rowconfigure(0, weight=1)

        self.console_text = tk.Text(console_frame, height=15, wrap=tk.WORD, bg='#252526', fg='#CCCCCC', insertbackground='#FFFFFF')
        self.console_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        scrollbar = ttk.Scrollbar(console_frame, orient=tk.VERTICAL, command=self.console_text.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.console_text.config(yscrollcommand=scrollbar.set)

        # Configure logging to console
        text_handler = TextHandler(self.console_text)
        text_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(text_handler)

    def create_groups_tab(self):
        """Create the Groups tab with sub-tabs."""
        frame = ttk.Frame(self.groups_tab, padding="20")
        frame.pack(fill='both', expand=True)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)

        # Sub-tabbed interface
        sub_notebook = ttk.Notebook(frame)
        sub_notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Sub-tabs
        self.add_group_tab = ttk.Frame(sub_notebook)
        self.add_members_tab = ttk.Frame(sub_notebook)
        self.remove_members_tab = ttk.Frame(sub_notebook)
        self.delete_group_tab = ttk.Frame(sub_notebook)
        self.remove_person_tab = ttk.Frame(sub_notebook)
        sub_notebook.add(self.add_group_tab, text="Add Group")
        sub_notebook.add(self.add_members_tab, text="Add Members")
        sub_notebook.add(self.remove_members_tab, text="Remove Members")
        sub_notebook.add(self.delete_group_tab, text="Delete Group")
        sub_notebook.add(self.remove_person_tab, text="Remove Person")

        # Initialize sub-tab contents
        self.create_add_group_subtab()
        self.create_add_members_subtab()
        self.create_remove_members_subtab()
        self.create_delete_group_subtab()
        self.create_remove_person_subtab()

    def create_add_group_subtab(self):
        """Create the Add Group sub-tab."""
        frame = ttk.LabelFrame(self.add_group_tab, text="Add New Group", padding="15", relief="groove")
        frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N), padx=10, pady=10)
        frame.columnconfigure(1, weight=1)

        ttk.Label(frame, text="Group Name:").grid(row=0, column=0, sticky=tk.W, pady=8)
        self.group_name_entry = ttk.Entry(frame)
        self.group_name_entry.grid(row=0, column=1, padx=5, pady=8, sticky=(tk.W, tk.E))
        ttk.Button(frame, text="Add Group", command=self.add_group).grid(row=1, column=0, columnspan=2, pady=10, sticky=tk.EW)

    def create_add_members_subtab(self):
        """Create the Add Members sub-tab."""
        frame = ttk.LabelFrame(self.add_members_tab, text="Add Members to Group", padding="15", relief="groove")
        frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=10, pady=10)
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)

        # Group Selection
        ttk.Label(frame, text="Select Group:").grid(row=0, column=0, sticky=tk.W, pady=8)
        self.add_members_group_var = tk.StringVar()
        self.add_members_group_combobox = ttk.Combobox(frame, textvariable=self.add_members_group_var, values=list(self.groups.keys()), state="readonly")
        self.add_members_group_combobox.grid(row=0, column=1, padx=5, pady=8, sticky=(tk.W, tk.E))
        self.add_members_group_combobox.bind('<<ComboboxSelected>>', self.update_add_members_people_listbox)

        # People List
        ttk.Label(frame, text="Available People:").grid(row=1, column=0, sticky=tk.W, pady=8)
        self.add_members_people_listbox = tk.Listbox(frame, selectmode=tk.MULTIPLE, height=12, width=30, bg='#252526', fg='#CCCCCC', selectbackground='#264F78')
        self.add_members_people_listbox.grid(row=1, column=1, padx=5, pady=8, sticky=(tk.W, tk.E))
        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=self.add_members_people_listbox.yview)
        scrollbar.grid(row=1, column=2, sticky=(tk.N, tk.S))
        self.add_members_people_listbox.config(yscrollcommand=scrollbar.set)

        # Add Members Button
        ttk.Button(frame, text="Add Members", command=self.add_members_to_group).grid(row=2, column=0, columnspan=2, pady=10, sticky=tk.EW)

        # Populate people list
        self.update_add_members_people_listbox(None)

    def create_remove_members_subtab(self):
        """Create the Remove Members sub-tab."""
        frame = ttk.LabelFrame(self.remove_members_tab, text="Remove Members from Group", padding="15", relief="groove")
        frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=10, pady=10)
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)

        # Group Selection
        ttk.Label(frame, text="Select Group:").grid(row=0, column=0, sticky=tk.W, pady=8)
        self.remove_members_group_var = tk.StringVar()
        self.remove_members_group_combobox = ttk.Combobox(frame, textvariable=self.remove_members_group_var, values=list(self.groups.keys()), state="readonly")
        self.remove_members_group_combobox.grid(row=0, column=1, padx=5, pady=8, sticky=(tk.W, tk.E))
        self.remove_members_group_combobox.bind('<<ComboboxSelected>>', self.update_remove_members_listbox)

        # Group Members
        ttk.Label(frame, text="Group Members:").grid(row=1, column=0, sticky=tk.W, pady=8)
        self.group_members_listbox = tk.Listbox(frame, selectmode=tk.MULTIPLE, height=12, width=30, bg='#252526', fg='#CCCCCC', selectbackground='#264F78')
        self.group_members_listbox.grid(row=1, column=1, padx=5, pady=8, sticky=(tk.W, tk.E))
        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=self.group_members_listbox.yview)
        scrollbar.grid(row=1, column=2, sticky=(tk.N, tk.S))
        self.group_members_listbox.config(yscrollcommand=scrollbar.set)

        # Remove Members Button
        ttk.Button(frame, text="Remove Members", command=self.remove_members_from_group).grid(row=2, column=0, columnspan=2, pady=10, sticky=tk.EW)

        # Populate members list
        self.update_remove_members_listbox(None)

    def create_delete_group_subtab(self):
        """Create the Delete Group sub-tab."""
        frame = ttk.LabelFrame(self.delete_group_tab, text="Delete Group", padding="15", relief="groove")
        frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N), padx=10, pady=10)
        frame.columnconfigure(1, weight=1)

        # Group Selection
        ttk.Label(frame, text="Select Group:").grid(row=0, column=0, sticky=tk.W, pady=8)
        self.delete_group_var = tk.StringVar()
        self.delete_group_combobox = ttk.Combobox(frame, textvariable=self.delete_group_var, values=list(self.groups.keys()), state="readonly")
        self.delete_group_combobox.grid(row=0, column=1, padx=5, pady=8, sticky=(tk.W, tk.E))

        # Delete Group Button
        ttk.Button(frame, text="Delete Group", command=self.delete_group).grid(row=1, column=0, columnspan=2, pady=10, sticky=tk.EW)

    def create_remove_person_subtab(self):
        """Create the Remove Person sub-tab."""
        frame = ttk.LabelFrame(self.remove_person_tab, text="Remove Person", padding="15", relief="groove")
        frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=10, pady=10)
        frame.columnconfigure(1, weight=1)

        # People List
        ttk.Label(frame, text="Available People:").grid(row=0, column=0, sticky=tk.W, pady=8)
        self.remove_person_listbox = tk.Listbox(frame, selectmode=tk.MULTIPLE, height=12, width=30, bg='#252526', fg='#CCCCCC', selectbackground='#264F78')
        self.remove_person_listbox.grid(row=0, column=1, padx=5, pady=8, sticky=(tk.W, tk.E))
        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=self.remove_person_listbox.yview)
        scrollbar.grid(row=0, column=2, sticky=(tk.N, tk.S))
        self.remove_person_listbox.config(yscrollcommand=scrollbar.set)

        # Remove Person Button
        ttk.Button(frame, text="Remove Person", command=self.remove_person).grid(row=1, column=0, columnspan=2, pady=10, sticky=tk.EW)

        # Populate people list
        self.update_remove_person_listbox()

    def create_predict_tab(self):
        """Create the Predict tab with console."""
        frame = ttk.Frame(self.predict_tab, padding="20")
        frame.pack(fill='both', expand=True)

        # Input Frame
        input_frame = ttk.LabelFrame(frame, text="Run Prediction", padding="15", relief="groove")
        input_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N), pady=10)
        input_frame.columnconfigure(1, weight=1)

        # Video Path
        ttk.Label(input_frame, text="Video Path:").grid(row=0, column=0, sticky=tk.W, pady=8)
        self.video_path_entry = ttk.Entry(input_frame)
        self.video_path_entry.grid(row=0, column=1, columnspan=2, padx=5, pady=8, sticky=(tk.W, tk.E))
        ttk.Button(input_frame, text="Browse", command=self.browse_video).grid(row=0, column=3, padx=10, pady=8)

        # Output Directory
        ttk.Label(input_frame, text="Output Directory:").grid(row=1, column=0, sticky=tk.W, pady=8)
        self.output_dir_entry = ttk.Entry(input_frame)
        self.output_dir_entry.grid(row=1, column=1, columnspan=2, padx=5, pady=8, sticky=(tk.W, tk.E))
        ttk.Button(input_frame, text="Browse", command=self.browse_output_dir).grid(row=1, column=3, padx=10, pady=8)
        if self.last_output_folder:
            self.output_dir_entry.insert(0, self.last_output_folder)

        # Group Selection
        ttk.Label(input_frame, text="Select Group:").grid(row=2, column=0, sticky=tk.W, pady=8)
        self.group_var = tk.StringVar()
        self.group_combobox = ttk.Combobox(input_frame, textvariable=self.group_var, values=list(self.groups.keys()), state="readonly")
        self.group_combobox.grid(row=2, column=1, padx=5, pady=8, sticky=tk.W)
        self.group_combobox.set("All")

        # Confidence Threshold
        ttk.Label(input_frame, text="Confidence Threshold:").grid(row=3, column=0, sticky=tk.W, pady=8)
        self.confidence_var = tk.StringVar(value="0.65")
        ttk.Entry(input_frame, textvariable=self.confidence_var, width=10).grid(row=3, column=1, sticky=tk.W, padx=5, pady=8)

        # Frame Threshold
        ttk.Label(input_frame, text="Frame Threshold:").grid(row=4, column=0, sticky=tk.W, pady=8)
        self.threshold_var = tk.StringVar(value="10")
        ttk.Entry(input_frame, textvariable=self.threshold_var, width=10).grid(row=4, column=1, sticky=tk.W, padx=5, pady=8)

        # Run Prediction Button
        ttk.Button(input_frame, text="Run Prediction", command=self.run_prediction).grid(row=5, column=0, columnspan=4, pady=15, sticky=tk.EW)

        # Console Output
        console_frame = ttk.LabelFrame(frame, text="Console Output", padding="10", relief="groove")
        console_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        console_frame.columnconfigure(0, weight=1)
        console_frame.rowconfigure(0, weight=1)

        self.predict_console_text = tk.Text(console_frame, height=15, wrap=tk.WORD, bg='#252526', fg='#CCCCCC', insertbackground='#FFFFFF')
        self.predict_console_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        scrollbar = ttk.Scrollbar(console_frame, orient=tk.VERTICAL, command=self.predict_console_text.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.predict_console_text.config(yscrollcommand=scrollbar.set)

        # Configure logging to predict console
        predict_text_handler = TextHandler(self.predict_console_text)
        predict_text_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(predict_text_handler)

    def browse_file(self):
        """Browse for image, video, or folder."""
        media_type = self.media_source_var.get()
        if media_type == "folder":
            path = filedialog.askdirectory(title="Select Folder")
        else:
            filetypes = [("Images", "*.jpg *.jpeg *.png *.bmp *.tiff")] if media_type == "image" else [("Videos", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv")]
            path = filedialog.askopenfilename(title=f"Select {media_type.capitalize()}", filetypes=filetypes)
        if path:
            self.file_entry.delete(0, tk.END)
            self.file_entry.insert(0, path)

    def browse_video(self):
        """Browse for video file."""
        video_path = filedialog.askopenfilename(title="Select Video", filetypes=[("Videos", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv")])
        if video_path:
            self.video_path_entry.delete(0, tk.END)
            self.video_path_entry.insert(0, video_path)

    def browse_output_dir(self):
        """Browse for output directory."""
        output_dir = filedialog.askdirectory(title="Select Output Directory")
        if output_dir:
            self.output_dir_entry.delete(0, tk.END)
            self.output_dir_entry.insert(0, output_dir)
            self.last_output_folder = output_dir
            self.save_groups()

    def add_person(self):
        """Add a person by processing media and generating embeddings."""
        person_name = self.person_name_entry.get().strip()
        media_path = self.file_entry.get().strip()
        media_type = self.media_source_var.get()
        quality_threshold = self.quality_var.get()

        if not person_name or not media_path:
            messagebox.showerror("Error", "Please provide a person name and select a media file/folder.")
            return

        try:
            quality_threshold = int(quality_threshold)
            if quality_threshold < 0 or quality_threshold > 100:
                raise ValueError("Quality threshold must be between 0 and 100")
        except ValueError:
            messagebox.showerror("Error", "Invalid quality threshold. Please enter a number between 0 and 100.")
            return

        logger.info(f"Adding person: {person_name}, Media: {media_path}, Type: {media_type}, Quality: {quality_threshold}")

        try:
            if media_type == "image":
                faces_saved, augmented_faces, faces_rejected, pose_counts = process_single_image(media_path, person_name, quality_threshold)
            elif media_type == "video":
                faces_saved, augmented_faces, faces_rejected, pose_counts = process_video(media_path, person_name, quality_threshold)
            else:
                faces_saved, augmented_faces, faces_rejected, pose_counts = process_person_folder(media_path, person_name, quality_threshold)

            if faces_saved == 0:
                messagebox.showwarning("Warning", f"No faces saved for {person_name}. Check input or quality threshold.")
                return

            generate_embeddings("face_images", "face_embeddings", person_name)
            if person_name not in self.groups["All"]:
                self.groups["All"].append(person_name)
                self.save_groups()
                self.update_add_members_people_listbox(None)
                self.update_remove_person_listbox()
                self.update_remove_members_listbox(None)
            messagebox.showinfo("Success", f"Added {person_name}: {faces_saved} faces saved, {augmented_faces} augmented, {faces_rejected} rejected")
        except Exception as e:
            logger.error(f"Failed to add person: {e}")
            messagebox.showerror("Error", f"Failed to add person: {e}")

    def add_group(self):
        """Add a new group."""
        group_name = self.group_name_entry.get().strip()
        if not group_name:
            messagebox.showerror("Error", "Please enter a group name.")
            return
        if group_name in self.groups:
            messagebox.showerror("Error", f"Group '{group_name}' already exists.")
            return
        self.groups[group_name] = []
        self.save_groups()
        self.update_comboboxes()
        messagebox.showinfo("Success", f"Group '{group_name}' added successfully.")

    def remove_person(self):
        """Remove selected people from the system."""
        selected_people = [self.remove_person_listbox.get(i) for i in self.remove_person_listbox.curselection()]
        if not selected_people:
            messagebox.showerror("Error", "Please select at least one person to remove.")
            return

        if not messagebox.askyesno("Confirm", f"Are you sure you want to remove {len(selected_people)} person(s)? This will delete their images and embeddings."):
            return

        try:
            for person in selected_people:
                person_dir = Path("face_images") / person
                if person_dir.exists():
                    shutil.rmtree(person_dir)
                    logger.debug(f"Deleted face_images directory for {person}")

                embedding_file = Path("face_embeddings") / f"{person}_embeddings.pt"
                if embedding_file.exists():
                    os.remove(embedding_file)
                    logger.debug(f"Deleted embedding file for {person}")

                for group_name in self.groups:
                    if person in self.groups[group_name]:
                        self.groups[group_name].remove(person)
                        logger.debug(f"Removed {person} from group {group_name}")

            self.save_groups()
            self.populate_people_list()
            self.update_add_members_people_listbox(None)
            self.update_remove_person_listbox()
            self.update_remove_members_listbox(None)
            self.update_comboboxes()
            messagebox.showinfo("Success", f"Removed {len(selected_people)} person(s) successfully.")
        except Exception as e:
            logger.error(f"Failed to remove person(s): {e}")
            messagebox.showerror("Error", f"Failed to remove person(s): {e}")

    def update_add_members_people_listbox(self, event):
        """Update the people listbox in Add Members sub-tab."""
        self.add_members_people_listbox.delete(0, tk.END)
        for person in self.groups["All"]:
            self.add_members_people_listbox.insert(tk.END, person)

    def update_remove_person_listbox(self):
        """Update the people listbox in Remove Person sub-tab."""
        self.remove_person_listbox.delete(0, tk.END)
        for person in self.groups["All"]:
            self.remove_person_listbox.insert(tk.END, person)

    def update_remove_members_listbox(self, event):
        """Update the group members listbox in Remove Members sub-tab."""
        self.group_members_listbox.delete(0, tk.END)
        group_name = self.remove_members_group_var.get()
        if group_name in self.groups:
            for member in self.groups[group_name]:
                self.group_members_listbox.insert(tk.END, member)

    def update_comboboxes(self):
        """Update all group comboboxes."""
        group_list = list(self.groups.keys())
        if self.add_members_group_combobox:
            self.add_members_group_combobox['values'] = group_list
        if self.remove_members_group_combobox:
            self.remove_members_group_combobox['values'] = group_list
        if self.delete_group_combobox:
            self.delete_group_combobox['values'] = group_list
        if self.group_combobox:
            self.group_combobox['values'] = group_list

    def add_members_to_group(self):
        """Add selected people to the selected group."""
        group_name = self.add_members_group_var.get()
        if not group_name:
            messagebox.showerror("Error", "Please select a group.")
            return
        if group_name not in self.groups:
            messagebox.showerror("Error", f"Group '{group_name}' does not exist.")
            return
        selected_people = [self.add_members_people_listbox.get(i) for i in self.add_members_people_listbox.curselection()]
        if not selected_people:
            messagebox.showerror("Error", "Please select at least one person to add.")
            return
        logger.debug(f"Adding people {selected_people} to group {group_name}")
        for person in selected_people:
            if person not in self.groups[group_name]:
                self.groups[group_name].append(person)
        self.save_groups()
        self.update_remove_members_listbox(None)
        messagebox.showinfo("Success", f"Added {len(selected_people)} people to {group_name}.")

    def remove_members_from_group(self):
        """Remove selected members from the selected group."""
        group_name = self.remove_members_group_var.get()
        if not group_name:
            messagebox.showerror("Error", "Please select a group.")
            return
        if group_name == "All":
            messagebox.showerror("Error", "Cannot remove members from 'All' group.")
            return
        selected_members = [self.group_members_listbox.get(i) for i in self.group_members_listbox.curselection()]
        if not selected_members:
            messagebox.showerror("Error", "Please select at least one member to remove.")
            return
        logger.debug(f"Removing members {selected_members} from group {group_name}")
        self.groups[group_name] = [m for m in self.groups[group_name] if m not in selected_members]
        self.save_groups()
        self.update_remove_members_listbox(None)
        messagebox.showinfo("Success", f"Removed {len(selected_members)} members from {group_name}.")

    def delete_group(self):
        """Delete the selected group."""
        group_name = self.delete_group_var.get()
        if not group_name:
            messagebox.showerror("Error", "Please select a group to delete.")
            return
        if group_name == "All":
            messagebox.showerror("Error", "Cannot delete 'All' group.")
            return
        del self.groups[group_name]
        self.save_groups()
        self.update_comboboxes()
        messagebox.showinfo("Success", f"Group '{group_name}' deleted successfully.")

    def run_prediction(self):
        """Run face recognition on the selected video."""
        video_path = self.video_path_entry.get().strip()
        output_dir = self.output_dir_entry.get().strip()
        group_name = self.group_var.get()
        confidence = self.confidence_var.get()
        threshold = self.threshold_var.get()

        if not video_path or not output_dir:
            messagebox.showerror("Error", "Please provide a video path and output directory.")
            return

        try:
            confidence = float(confidence)
            if confidence < 0 or confidence > 1:
                raise ValueError("Confidence threshold must be between 0 and 1")
            threshold = int(threshold)
            if threshold < 0:
                raise ValueError("Frame threshold must be non-negative")
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid input: {e}")
            return

        if group_name not in self.groups:
            messagebox.showerror("Error", f"Group '{group_name}' does not exist.")
            return

        people = self.groups[group_name] if group_name != "All" else None
        logger.info(f"Running prediction: Video={video_path}, OutputDir={output_dir}, Group={group_name}, Confidence={confidence}, Threshold={threshold}")

        try:
            # Generate custom filename with date_hour and random number
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            random_num = random.randint(10000, 99999)
            output_filename = f"{timestamp}_R{random_num}"
            desired_output_path = str(Path(output_dir) / f"{output_filename}.mp4")

            # Run prediction with default output filename
            person_counts, temp_output_path = predict_video(video_path, output_dir=output_dir, confidence_threshold=confidence, people=people)
            if not person_counts:
                messagebox.showerror("Error", "Prediction failed. Check console for details.")
                return

            # Rename the output video to desired format
            if temp_output_path != desired_output_path and os.path.exists(temp_output_path):
                os.rename(temp_output_path, desired_output_path)
                logger.debug(f"Renamed video from {temp_output_path} to {desired_output_path}")
            output_path = desired_output_path

            # Save attendance Excel with matching filename
            attendance_df = pd.DataFrame([
                {"Person": person, "Frames Detected": count, "Status": "present" if count >= threshold else "absent"}
                for person, count in person_counts.items()
            ])
            output_excel = Path(output_path).with_name(f"{output_filename}_attendance.xlsx")
            attendance_df.to_excel(output_excel, index=False)

            messagebox.showinfo("Success", f"Prediction complete. Output: {output_path}\nAttendance Excel: {output_excel}")
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            messagebox.showerror("Error", f"Prediction failed: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = AttendanceApp(root)
    root.mainloop()