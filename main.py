import json
import os
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk

import faiss
import numpy as np
import pandas as pd
import requests
import search_utils
import torch
from PIL import Image, ImageTk, ImageFilter

# My modules
from model_clip import Model
from sklearn.decomposition import PCA
import features_extractor_compressed as fec
import net_requests as nr

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

shown_images = 156
displayed_images_indices = []
selected_images_indices = []
displayed_images = []  # images that are being shown currently
images_buttons = []  # holds the Tkinter button widgets that display the images
selected_images = []  # keep track of images that have been selected by the user
url = "" # url to evaluation server
dataset_path = "" # path to dataset of images

# storing all pathes to all files in the DB
filenames = []
for file_name in sorted(os.listdir(dataset_path)):
    if file_name.endswith(('.png', '.jpg', '.jpeg')):  # Ensure only expected file types are added
        filename_full = os.path.join(dataset_path, file_name)
        filenames.append(filename_full)

model = Model(display_size=shown_images)
features = model.features
features_np = features.numpy()

selected_positive_indices = selected_images_indices
selected_negative_indices = []

cumulative_importance_scores = [np.ones(features_np.shape[1])]

# SERVER COMMUNICATION (if needed, you can turn it off)

# Read json configuration
with open('config.json') as f:
    config = json.load(f)
    username = config['username']
    password = config['password']

metadata = pd.read_csv("metadata.csv")

session_id, competition_id, competition_name = nr.get_session_id_for_user(username, password)
print("***")
print(f"Session id: {session_id}\nCompetition: {competition_name} ({competition_id})")
print("***")

# FUNCTIONS THAT SUPPORT GUI

def update_display(indices):
    # i is numerator and index is position in the file list 5765
    displayed_images_indices.clear()
    for i, index in enumerate(indices):
        if i < shown_images:
            displayed_images_indices.append(index)
            image_path = filenames[index]
            image = Image.open(image_path).resize(image_size)
            displayed_images[i] = ImageTk.PhotoImage(image)
            images_buttons[i].configure(image=displayed_images[i], text=image_path)
            images_buttons[i].image = displayed_images[i]  # Save a reference
            # images_buttons[i].bind('<Button-3>', lambda event, path=image_path:open_image_external(path))
            images_buttons[i].bind('<Button-3>', lambda event, path=image_path: zoom_image(path))
    hide_borders()
    # print("Display:", displayed_images_indices)
    # print("Selected:", selected_images)


def search_clip(query):
    indices = model.search_clip(query)
    # print(indices)
    hide_borders()
    update_display(indices)


def on_double_click():
    hide_borders()


def close_win(e):
    root.destroy()


def on_click(button_index):
    # Check if button_index is valid for displayed_images_indices
    if button_index < len(displayed_images_indices):
        real_index = displayed_images_indices[button_index]  # Map to the real index

        # Toggle selection state based on the button's current background color
        if images_buttons[button_index].cget("bg") == "yellow":  # Already selected
            images_buttons[button_index].config(bg="black")  # Deselect
            selected_images.remove(images_buttons[button_index].cget("text"))
            if real_index in selected_images_indices:
                selected_images_indices.remove(real_index)  # Remove from the tracking list
        else:  # Not selected
            images_buttons[button_index].config(bg="yellow")  # Select
            selected_images.append(images_buttons[button_index].cget("text"))
            if real_index not in selected_images_indices:
                selected_images_indices.append(real_index)  # Add to the tracking list

        print(f"Selected image indices: {selected_images_indices}")
        update_last_selected_label()
        global selected_negative_indices
        selected_negative_indices = search_utils.update_negative_indices(displayed_images_indices,
                                                                         selected_images_indices)
    else:
        print(f"Index {button_index} out of range for images_buttons list.")


def update_display_based_on_distance(index_mode="cosine"):
    global selected_positive_indices, selected_negative_indices, features_np, index

    index = search_utils.setup_faiss_index(features_np, index_mode)
    # refined_indices = search_utils.refine_search(selected_positive_indices, selected_negative_indices, features, index, shown_images)
    refined_indices = search_utils.refine_search_v3(selected_positive_indices, selected_negative_indices, features_np,
                                                    index, shown_images)
    print("Refined indices:", refined_indices)
    update_display(refined_indices)


def hide_borders(hide_selection=True):
    global selected_images
    for button in images_buttons:
        button.config(bg="black")
    if hide_selection:
        selected_images = []
    selected_images_indices.clear()

# only for debugging purposes, you can select any image needed from database and test how simialrity search works.
def upload_and_search_image():

    file_paths = filedialog.askopenfilenames(
        filetypes=[("Image files", "*.jpg *.jpeg *.png")])  # Allows selecting multiple files
    if file_paths:
        image_files = list(file_paths)  # Converts the tuple of file paths to a list
        image_files.sort()
        output_file = 'my_features_selected.npy'

        fec.encode_images_in_batches_from_list(image_files, batch_size=10, output_file=output_file)
        selected_tensors = fec.load_features(output_file)

        query_feature = torch.mean(selected_tensors, dim=0)
        query_feature = torch.nn.functional.normalize(query_feature.unsqueeze(0), p=2)
        query_feature_numpy = query_feature.numpy()

        print("Mean:", query_feature_numpy.shape)

        dimension = features_np.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Using Inner Product for Cosine Similarity
        features_norm = features_np.copy()
        faiss.normalize_L2(features_norm)
        index.add(features_norm)

        distances, indices = index.search(query_feature_numpy, shown_images)  # Search in the index
        print(indices.squeeze())
        update_display(indices.squeeze())


def update_last_selected_label():
    if selected_images_indices:
        last_selected = selected_images_indices[-1]  # Get the last item
        text_index.config(text=f"Last selected image: {last_selected}")
    else:
        text_index.config(text="Last selected image: None")  # No items selected


def open_image_external(filepath):
    if os.name == 'nt':  # For Windows
        os.startfile(filepath)
    elif os.name == 'posix':  # For Unix-like systems
        subprocess.run(['xdg-open', filepath])


def zoom_image(image_path):
    new_window = tk.Toplevel(root)
    new_window.title("Zoomed Image")
    img = Image.open(image_path)

    # Use the new resampling method
    img = img.resize((img.width * 4, img.height * 4), Image.Resampling.LANCZOS)

    # Apply a sharpening filter
    img = img.filter(ImageFilter.SHARPEN)

    photo = ImageTk.PhotoImage(img)
    label = tk.Label(new_window, image=photo)
    label.image = photo  # keep a reference!
    label.pack()


# IMPLEMENTIN GUI

root = tk.Tk()
root.title("My Pretty Little Clip Model")
style = ttk.Style()
style.theme_use('alt')

image_size = (int(root.winfo_screenwidth() / 14) - 4, int(root.winfo_screenheight() / 8) - 6)

# Main window configuration
window = ttk.Panedwindow(root, orient=tk.HORIZONTAL)
window.pack(fill=tk.BOTH, expand=True)

# Setup frames for search bar and results
search_bar = ttk.Frame(window, width=root.winfo_screenwidth() / 4, height=root.winfo_screenheight(), relief=tk.SUNKEN)
result_frame = ttk.Frame(window, width=(3 * root.winfo_screenwidth()) / 4, height=root.winfo_screenheight(),
                         relief=tk.SUNKEN)
window.add(search_bar, weight=1)
window.add(result_frame, weight=14)

# Create a canvas and a scrollbar in result_frame
canvas = tk.Canvas(result_frame)
scrollbar = ttk.Scrollbar(result_frame, orient="vertical", command=canvas.yview)
scrollable_frame = ttk.Frame(canvas)

# Scrollable frame configuration
scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(
        scrollregion=canvas.bbox("all")
    )
)

canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)

# Packing canvas and scrollbar
canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")


# Function to handle mouse wheel
def _on_mouse_wheel(event):
    canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")


# Bind the mouse wheel event
root.bind_all("<MouseWheel>", _on_mouse_wheel)

# Now create buttons within scrollable_frame instead of result_frame
images_buttons = []  # Resetting the list to ensure it's empty before populating
displayed_images = []
for s in range(shown_images):
    # load image
    img_path = filenames[s] if s < len(filenames) else filenames[0]  # Just a fallback in case of fewer files
    img = Image.open(img_path).resize(image_size)
    photo = ImageTk.PhotoImage(img)
    displayed_images.append(photo)

    # create button
    image_button = tk.Button(scrollable_frame, image=photo, bg="black", bd=2, text=img_path,
                             command=(lambda j=s: on_click(j)))
    image_button.grid(row=(s // 12), column=(s % 12), sticky=tk.W)
    images_buttons.append(image_button)

# Remaining GUI setup
tk.Label(search_bar, text="Text query:", font=('Arial', 12)).pack(side=tk.TOP, pady=10)
text_input = ttk.Entry(search_bar, width=32)
text_input.bind("<Return>", (lambda event: search_clip(text_input.get())))
text_input.pack(side=tk.TOP, pady=10)

clip_button = tk.Button(search_bar, text="Search Clip", command=lambda: search_clip(text_input.get()))
clip_button.pack(side=tk.TOP)
update_button = tk.Button(search_bar, text="Update Display HNSW",
                          command=lambda: update_display_based_on_distance("hnsw"))
update_button.pack(side=tk.TOP, pady=10)

update_button = tk.Button(search_bar, text="Update Display IVF",
                          command=lambda: update_display_based_on_distance("ivf"))
update_button.pack(side=tk.TOP, pady=10)

update_button_strict = tk.Button(search_bar, text="Update Display Cosine",
                                 command=lambda: update_display_based_on_distance("cosine"))
update_button_strict.pack(side=tk.TOP, pady=10)

update_button_strict = tk.Button(search_bar, text="Update Display Inner Product",
                                 command=lambda: update_display_based_on_distance("product"))
update_button_strict.pack(side=tk.TOP, pady=10)

# add info labels
tk.Label(search_bar, text="Find index (1-23064):").pack(side=tk.TOP, pady=10)
text_index = tk.Label(search_bar, text="Last selected image: ")
text_index.pack(side=tk.TOP, pady=5)
text_answer = tk.Label(search_bar, text="Last submission status: None")
text_answer.pack(side=tk.TOP, pady=10)

# sending select result
send_result_b = tk.Button(search_bar, text="Send selected index", command=(
    lambda: nr.send_result(selected_images, text_answer, session_id, competition_id, competition_name, metadata)))
send_result_b.pack(side=tk.TOP, pady=5)
# set control-v to set result
root.bind('<Control-v>', lambda e: send_result())

root.mainloop()
