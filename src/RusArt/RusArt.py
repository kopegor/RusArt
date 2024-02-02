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

MODEL_WEIGHTS = "./swin_t_last.pt"


img_mode = st.radio(
    "Select photo mode",
    ["Take photo", "Upload file"],
    horizontal=True
)

image = None

if img_mode == "Upload file":
    file = st.file_uploader('Upload file')
    if file:
        image = file.getvalue()
        image = np.array(Image.open(io.BytesIO(image)))
else:
    photo = st.camera_input('Take photo')
    if photo:
        image = photo.getvalue()
        image = np.array(Image.open(io.BytesIO(image)))

if image is not None:
    image = Image.fromarray(image)
    st.image(image)

    art_type = get_category(image, MODEL_WEIGHTS)
    st.write(f"***Тип произведения искуссства на фото***: {art_type}")





