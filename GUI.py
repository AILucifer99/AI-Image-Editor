from PIL import Image
from torch.optim import Adam
from torch.utils.data import DataLoader
from RealESRGAN import RealESRGAN
from torchvision import datasets
from torchvision import transforms
from tqdm import tqdm as tqdm
from collections import namedtuple
from torchvision import models
import numpy as np
import os
import glob
import streamlit as st
import io
import argparse
import sys
import time
import re
import torch.onnx
import torch
import authentication.authentication as auth  

# Define a secure username and password
VALID_USERNAME = auth.user_name
VALID_PASSWORD = auth.password

# Create a session state variable
session_state = st.session_state

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_scale_factor = 2  # Allowed Values are 2, 4, and 8
result_folder = "super-resolution-result"


# Check if the user is logged in
def is_logged_in():
    return session_state.get('logged_in', False)

# Login function
def login(username, password):
    if username == VALID_USERNAME and password == VALID_PASSWORD:
        session_state['logged_in'] = True
        st.success("Logged in successfully!")
    else:
        st.error("Invalid username or password.")

# Logout function
def logout():
    session_state['logged_in'] = False
    st.success("Logged out successfully!")


#################################################################################
#################### CODE FOR THE SUPER RESOLUTION ON IMAGES ####################
#################################################################################

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


def preprocessImage(input_path, model_scale_factor):
    result_image_path = os.path.join(result_folder, os.path.basename(input_path))
    image = Image.open(input_path).convert("RGB")
    model = loadUpcalingModel(model_location=device, scale_factor=model_scale_factor)
    sr_image = model.predict(np.array(image))
    sr_image.save(result_image_path)
    print("Super Resolution completed successfully....")
    print("Result saved at: {}".format(result_image_path))
    return result_image_path


def main(input_path, super_resolution_scale_factor=2):
    createFolders()
    result_image_path = preprocessImage(input_path, super_resolution_scale_factor)
    return result_image_path


# Super Resolution UI Controller
def imageSuperResolution() :
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


#################################################################################
#################### CODE FOR THE STYLE TRANSFER ON IMAGES ######################
#################################################################################
class TransformerNet(torch.nn.Module):
    def __init__(self):
        super(TransformerNet, self).__init__()
        # Initial convolution layers
        self.conv1 = ConvLayer(3, 32, kernel_size=9, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(32, affine=True)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.in2 = torch.nn.InstanceNorm2d(64, affine=True)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.in3 = torch.nn.InstanceNorm2d(128, affine=True)
        # Residual layers
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)
        # Upsampling Layers
        self.deconv1 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2)
        self.in4 = torch.nn.InstanceNorm2d(64, affine=True)
        self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
        self.in5 = torch.nn.InstanceNorm2d(32, affine=True)
        self.deconv3 = ConvLayer(32, 3, kernel_size=9, stride=1)
        # Non-linearities
        self.relu = torch.nn.ReLU()

    def forward(self, X):
        y = self.relu(self.in1(self.conv1(X)))
        y = self.relu(self.in2(self.conv2(y)))
        y = self.relu(self.in3(self.conv3(y)))
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)
        y = self.relu(self.in4(self.deconv1(y)))
        y = self.relu(self.in5(self.deconv2(y)))
        y = self.deconv3(y)
        return y


class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual
        return out


class UpsampleConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = torch.nn.functional.interpolate(x_in, mode='nearest', scale_factor=self.upsample)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out


class Vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out


def load_image(filename, size=None, scale=None):
    img = Image.open(filename)
    if size is not None:
        img = img.resize((size, size), Image.ANTIALIAS)
    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
    return img


def save_image(filename, data):
    img = data.clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    img = Image.fromarray(img)
    img.save(filename)


def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


def normalize_batch(batch):
    # normalize using imagenet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    batch = batch.div_(255.0)
    return (batch - mean) / std


@st.cache
def load_model(model_path):
    print('load model')
    with torch.no_grad():
        style_model = TransformerNet()
        state_dict = torch.load(model_path)
        # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
        for k in list(state_dict.keys()):
            if re.search(r'in\d+\.running_(mean|var)$', k):
                del state_dict[k]
        style_model.load_state_dict(state_dict)
        style_model.to(device)
        style_model.eval()
        return style_model


@st.cache
def stylize(style_model, content_image, output_image):
    content_image = load_image(content_image)
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = style_model(content_image).cpu()
            
    save_image(output_image, output[0])


# Style Transfer UI Controller
def styleTransferStreamlit() :
    st.title('AI Image Style Transfer')
    st.sidebar.title("Stylization Options")

    img = st.sidebar.selectbox(
        'Select Image',
        ('Amber.jpg', 'Anime.jpg', 'Cat.png', 'Dog.jpg', 
        'Rastrapati-Bhavan.jpg', 'Taj-Mahal.jpg', 'other.jpg')
    )

    style_name = st.sidebar.selectbox(
        'Select Style',
        ('candy', 'mosaic', 'rain_princess', 'udnie')
    )

    model = "saved_models/" + style_name + ".pth"
    input_image = "images/content-images/" + img
    output_image = "images/output-images/" + style_name + "-" + img

    st.write('### Source image:')
    image = Image.open(input_image)
    st.image(image, width=600) # image: numpy array

    clicked = st.button('Stylize')

    if clicked:
        with st.spinner(text="Stylization Ongoing.....") :
            model = load_model(model)
            stylize(model, input_image, output_image)
        st.success("Stylization Completed.")

        st.write('## AI Generated Copyright Free Output image:')
        image = Image.open(output_image)
        st.image(image, width=600)


# Combined GUI Controller
def StreamlitCombinedWebUI(parse_UI) :
    if parse_UI :
        page_names_to_funcs = {
            "Image Style Synthesizer": styleTransferStreamlit,
            "Upscaling Image": imageSuperResolution,
        }
        demo_name = st.sidebar.selectbox("Choose the Pipeline", page_names_to_funcs.keys())
        page_names_to_funcs[demo_name]()


# Main application code
def PipeLineFunction(start=False):
    if start :
        st.title('AI Image Editing Web Application')

        # Check if the user is logged in
        if is_logged_in():
            st.write(f"Welcome, {auth.user_name}!")
            # Your style transfer code or any other authenticated functionality here
            StreamlitCombinedWebUI(True)
            st.button("Logout", on_click=logout)
        else:
            st.write("Please log in to access the application.")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            if st.button("Login", on_click=login, args=(username, password)):
                pass


if __name__ == "__main__" :
    PipeLineFunction(True)
