import cv2
import trimesh
import os.path as osp
import numpy as np
import torch
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET
from icecream import ic



# class ContDataset(Dataset):