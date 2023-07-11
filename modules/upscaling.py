from RealESRGAN import RealESRGAN
from PIL import Image
import numpy as np
import torch
import os
import glob
from tqdm import tqdm as tqdm
import streamlit as st
import io
import time


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_scale_factor = 2  # Allowed Values are 2, 4, and 8
result_folder = "super-resolution-result"


@st.cache
def loadUpcalingModel(model_location, scale_factor=2):
    print("Using the Device as: {}".format(device))
    print("Loading the Super Resolution GAN Model, please wait a while....")
    model = RealESRGAN(device, scale=model_scale_factor)
    model.load_weights(f"weights/RealESRGAN_x{model_scale_factor}.pth")
    print("Model loaded successfully to the device: {}".format(device))
    return model


def createFolders():
    os.makedirs(result_folder, exist_ok=True)


def preprocessImage(input_path):
    result_image_path = os.path.join(result_folder, os.path.basename(input_path))
    image = Image.open(input_path).convert("RGB")
    model = loadUpcalingModel(model_location=device, scale_factor=2)
    sr_image = model.predict(np.array(image))
    sr_image.save(result_image_path)
    print("Super Resolution completed successfully....")
    print("Result saved at: {}".format(result_image_path))
    return result_image_path


def main(input_path):
    createFolders()
    result_image_path = preprocessImage(input_path)
    return result_image_path


def imageSuperResolution(parse_pipeline=False) :
    if parse_pipeline :

        st.title("AI Image Super Resolution")
        # File uploader
        uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

        if uploaded_file is not None:
            with st.spinner(text="Super Resolution Ongoing.....") :
                # Save the uploaded file locally
                image_path = os.path.join(result_folder, uploaded_file.name)
                with open(image_path, "wb") as file:
                    file.write(uploaded_file.getbuffer())

                # Process the uploaded image
                result_image_path = main(image_path)
            st.success("Super Resolution Completed.")

            # Display the result image
            st.image(result_image_path, caption="Super Resolved Image")

            # Download button
            with open(result_image_path, "rb") as file:
                result_image_bytes = file.read()
            st.download_button(
                label="Download Super Resolved Image",
                data=result_image_bytes,
                file_name="super_resolved_image.jpg",
                mime="image/jpeg"
            )


if __name__ == "__main__":
    imageSuperResolution(True)
    
