# AI-Image-Editor
An implementation of an end to end image stylization and upscaling web application developed for the purpose of image editing. 

## Usage Guidelines
For running the Web GUI follow the steps properly and in a sequential manner.

1.   Clone the repository using the command `git clone https://github.com/AILucifer99/AI-Image-Editor`
2.   Once cloaning is completed, enter inside the folder using command `cd AI-Image-Editor`
3.   Install the primary dependencies that are require for the system to work. The libraries are `torch`, `torchvision`, `numpy`, `glob`, `streamlit`, `RealESRGAN`, the version of pytorch must be compatible with the `CUDA` for faster inference. The system was developed on a PC with a GPU provided by `Nvidia 3070 Ti` and `Intel Core-i9 Processor`.
4.   Once the libraries are installed successfully, then just run the command `streamlit run GUI.py`, and the WebUI will launch via a localhost.
5.   Login to the system using the provided credentials, `username - admin` and `password - admin@1234`. These credentials are provided inside the `authentication` folder.
6.   Once login is successfully completed, the WebUI will be redirected to the actual AI Image Editing page.
7.   Firstly, select the style that is to be applied on the image and just follow the process. 

