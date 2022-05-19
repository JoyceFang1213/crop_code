# import package 
import glob
import torch
from torch import optim
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
import PIL
from PIL import Image
import PIL.ImageOps
import PIL.ImageEnhance
import PIL.ImageDraw
import random
import math
import numpy as np
import pandas as pd
import torchvision
import matplotlib.pyplot as plt
import opendatasets as od
from torchsummary import summary
from functools import partial
import timm
from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.loss import LabelSmoothingCrossEntropy
from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
from timm.data.auto_augment import rand_augment_transform
from timm.data.transforms import RandomResizedCropAndInterpolation
from torchvision.transforms import InterpolationMode
from timm.scheduler import CosineLRScheduler
from sklearn.metrics import confusion_matrix

from sklearn.metrics import f1_score, precision_score
from sklearn.metrics import classification_report

from tqdm import tqdm

stats = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

test_tfms = transforms.Compose([
    transforms.Resize([384, 384]),
    transforms.ToTensor(),
    transforms.Normalize(*stats)
])


class FoldDataset(Dataset):
    def __init__(self, root, transform):
        self.data = glob.glob(f"{root}/*.jpg")
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img = Image.open(self.data[idx]).convert("RGB")
        img = self.transform(img)
        return img
        


test_dataset_new = FoldDataset("/work/u4952464/crop/final_test/test_dataset", test_tfms)
test_dataset = ImageFolder("/work/u4952464/crop/train")
test_loader = DataLoader(dataset=test_dataset_new, 
                         batch_size=16,
                         shuffle=False, 
                         num_workers=16, 
                         pin_memory=True, 
                         drop_last=False)
print(len(test_dataset_new))
idx_to_cls = {
    idx: cls for cls, idx in test_dataset.class_to_idx.items()
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class mymodel(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = timm.create_model('swin_base_patch4_window12_384_in22k', pretrained=True, num_classes=14)
    def forward(self, x):
        x = self.vit(x)
        return x

model = nn.DataParallel(mymodel())
# model = torch.load("vit_large_patch16_224_99.35.pth")
model.load_state_dict(torch.load("./final.pth"))
# model.load_state_dict()
model = model.to(device)

file_name = []
for x in test_dataset_new.data:
    tuple_name = x.replace("/work/u4952464/crop/final_test/test_dataset", "")
    file_name.append(tuple_name.split("/")[-1])

category = []
# true_labels = []
model.eval()

with torch.no_grad():
    for test_image in tqdm(test_loader):
        test_image = test_image.to(device)
        
        test_output = model(test_image).argmax(-1)
        test_output = test_output.tolist()
        category.extend(test_output)
#         true_labels.extend(label.tolist())

str_labels = [idx_to_cls[idx] for idx in category]
submit = pd.DataFrame({"image_filename": file_name, "label": str_labels})  

submit.to_csv(r'final.csv', index=False)

# weighted_precision_score = precision_score(true_labels, category, average="weighted")

# with open("acc.txt", "a") as f:
#     f.write(f"WP: {weighted_precision_score}\n")

# weighted_f1_score = f1_score(true_labels, category, average="weighted")

# with open("acc.txt", "a") as f:
#     f.write(f"F1: {weighted_f1_score}\n")

# print(weighted_precision_score)
# print(weighted_f1_score)

# result = classification_report(true_labels, category, output_dict=True)
# report = pd.DataFrame(result).T
# report.index.name = "class"
# report.to_csv("report.csv",
#               float_format="%.3f")

