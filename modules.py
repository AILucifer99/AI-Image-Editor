from RealESRGAN import RealESRGAN
from PIL import Image
import numpy as np
import os
import glob
from tqdm import tqdm as tqdm
import streamlit as st
import io
import argparse
import sys
import time
import re
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import torch.onnx
import torch
from collections import namedtuple
from torchvision import models
import authentication.authentication as auth  