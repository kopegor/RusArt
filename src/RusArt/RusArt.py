import numpy as np
import torch
from torchvision import transforms
import torchvision
import streamlit as st

from utils import init_model, ArtDataset, get_category

from PIL import Image
import cv2
import pandas as pd
import io

# path to trained model weights
MODEL_WEIGHTS = "./weights/swin_t_last.pt"

# mode of get image
img_mode = st.radio(
    "Select image mode",
    ["Take photo", "Upload image"],
    horizontal=True
)

# init image
image = None

if img_mode == "Upload image":
    # select image file
    file = st.file_uploader("Upload file")
    if file:
        # extract image
        image = file.getvalue()
        image = np.array(Image.open(io.BytesIO(image)))
else:
    # take photo from web camera
    photo = st.camera_input("Take photo")
    if photo:
        # extract photo
        image = photo.getvalue()
        image = np.array(Image.open(io.BytesIO(image)))

# check if image was extracted
if image is not None:
    # modify to PIL.Image from numpy array
    image = Image.fromarray(image)

    # display image
    st.image(image)

    # get predicted art category of image
    art_type = get_category(image, MODEL_WEIGHTS)
    st.write(f"***Тип произведения искуссства на фото***: {art_type}")
