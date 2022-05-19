# %%
# import package 
import PIL
from PIL import Image, ImageFile
import PIL.ImageOps
import PIL.ImageEnhance
import PIL.ImageDraw
import torch
from torch import optim
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import ImageFolder
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
from tqdm import tqdm 

# %%

ImageFile.LOAD_TRUNCATED_IMAGES = True

num_classes = 14


# %%


# hyperparameter
batch_size = 128

epochs = 35

lr = 0.0001
warmup_epoch = 5


# %%


rand_aug = rand_augment_transform(
    config_str="rand-m9-mstd0.5-inc1",
    hparams=dict(
        translate_const=int(384 * 0.45),
        img_mean=tuple([min(255, round(255 * x)) for x in (0.5071, 0.4867, 0.4408)]),
    )
)


# %%


# Use when training, don't put it in transforms
mixup_args = {
        'mixup_alpha': 0.3,
        'cutmix_alpha': 0.4,
        'cutmix_minmax': None,
        'prob': 0.7,
        'switch_prob': 0.,
        'mode': 'batch',
        'label_smoothing': 0,
        'num_classes': num_classes
    }
mixup_cutmix = Mixup(
    **mixup_args
)


# %%


# Create transforms
stats = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

random_transform = RandomResizedCropAndInterpolation(size=384, scale=(0.8, 1.0))
random_transform.interpolation = InterpolationMode.BILINEAR

train_tfms = transforms.Compose([
    random_transform,
    rand_aug,
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(*stats),
])

test_tfms = transforms.Compose([
    transforms.Resize([384, 384]),
    transforms.ToTensor(),
    transforms.Normalize(*stats)
])


# %%


# Create ImageFolder

train_dataset = ImageFolder(
                root="/work/u4952464/crop/train",
                transform=train_tfms)

test_dataset = ImageFolder(
                root="/work/u4952464/crop/test",
                transform=test_tfms)


# %%


# Create Dataloder
train_loader = DataLoader(dataset=train_dataset, 
                          batch_size=batch_size, 
                          shuffle=True, 
                          num_workers=16, 
                          pin_memory=True, 
                          drop_last=True)

test_loader = DataLoader(dataset=test_dataset, 
                         batch_size=batch_size,
                         shuffle=False, 
                         num_workers=16, 
                         pin_memory=True, 
                         drop_last=False)


# %%


# Create device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# %%
class mymodel(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = timm.create_model('swin_base_patch4_window12_384_in22k', pretrained=True, num_classes = 14)
    def forward(self, x):
        x = self.vit(x)
        return x

# model = deit_tiny_patch16_224(pretrained=True).to(device)

# model = nn.Sequential(nn.Upsample(size=(224,224), mode='bilinear'), model)
# model = 
model = mymodel()

# %%
# model.head =  nn.Linear(192, num_classes)


# %%


print(model)


# %%


model = model.to(device)
model = torch.nn.DataParallel(model)
model.load_state_dict(torch.load("weight/swin_base_patch4_window12_384_in22k_99.76.pth"))
# %%


def adjust_learning_rate(optimizer, epoch):
    learn_rate = lr
    if epoch < warmup_epoch:
        learn_rate = learn_rate / (warmup_epoch - epoch)
    else:
        learn_rate *= 0.5 * (1. + math.cos(math.pi * (epoch - warmup_epoch) / (epochs - warmup_epoch)))

    for param_group in optimizer.param_groups:
        param_group['lr'] = learn_rate


# %%


optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
# optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, lr=lr)

count_cls = {}
for cls in train_dataset.targets:
    if cls not in count_cls:
        count_cls[cls] = 1
    else:
        count_cls[cls] += 1

count_cls = dict(sorted(list(count_cls.items()), key=lambda x: x[0]))
count_cls = list(count_cls.values())

# Normalize
most_cls = max(count_cls)
count_cls = torch.tensor([most_cls / num for num in count_cls]).to(device)

criterion = nn.CrossEntropyLoss(weight=count_cls)


# %%


noise_args = dict(
    noise_range_t=None,
    noise_pct=0.67,
    noise_std=1.,
    noise_seed=42
)

lr_scheduler = CosineLRScheduler(
    optimizer,
    t_initial=epochs,
    t_mul=1.0,
    lr_min=1e-5,
    decay_rate=0.1,
    warmup_lr_init=5e-5,
    warmup_t=3,
    cycle_limit=1,
    t_in_epochs=True,
    **noise_args,
)

# %%


def get_lr_per_epoch(scheduler, num_epoch):
    lr_per_epoch = []
    for epoch in range(num_epoch):
        lr_per_epoch.append(scheduler.get_epoch_values(epoch))
    return lr_per_epoch

lr_per_epoch = get_lr_per_epoch(lr_scheduler, epochs)
plt.plot([i for i in range(epochs)], lr_per_epoch, label="With warmup");


test_scheduler = CosineLRScheduler(optimizer, t_initial=epochs)
lr_per_epoch = get_lr_per_epoch(test_scheduler, epochs)
plt.plot([i for i in range(epochs)], lr_per_epoch, label="Without warmup", alpha=0.8)

plt.legend()


# %%


total_train_loss_history = []
train_accuracy_history = []
total_test_loss_history = []
test_accuracy_history = []
best_accuracy = 0


# %%


for epoch in range(epochs):
    total_train_loss = 0
    train_accuracy = 0
    total_test_loss = 0
    test_accuracy = 0

    steps = 0
    total_steps = len(train_loader)
    model.train()
    adjust_learning_rate(optimizer, epoch)
    num_updates = (epoch) * len(train_loader)
    for image, label in tqdm(train_loader):
        steps += 1
        image = image.to(device)
        label = label.to(device)
        
        image, label = mixup_cutmix(image, label)

        output = model(image)
        label = label.argmax(-1)
        
        optimizer.zero_grad()
        train_loss = criterion(output, label)
        train_loss.backward()
        # nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()

        total_train_loss += train_loss.item() 
        prediction = output.argmax(-1) == label
        train_accuracy += prediction.sum().item() / label.size(0)

        # lr_scheduler.step_update(num_updates, None)

    # lr_scheduler.step(epoch + 1, None)

    model.eval()
    with torch.no_grad():
        for test_image, test_label in test_loader:
            test_image = test_image.to(device)
            test_label = test_label.to(device)
            
            test_output = model(test_image)
            test_loss = criterion(test_output, test_label)
            
            total_test_loss += test_loss.item()
            test_prediction = test_output.argmax(-1) == test_label
            test_accuracy += test_prediction.sum().item() / test_label.size(0)

    total_train_loss_history.append(total_train_loss / len(train_loader))
    total_test_loss_history.append(total_test_loss / len(test_loader))
    train_accuracy_history.append(train_accuracy / len(train_loader) * 100)
    test_accuracy_history.append(test_accuracy / len(test_loader) * 100)

    print("Epoch {}".format(epoch+1))
    print("Train loss : {}".format(total_train_loss / len(train_loader)))
    print("Test loss : {}".format(total_test_loss / len(test_loader)))
    print("Train accuracy : {}".format(train_accuracy / len(train_loader) * 100))
    print("Test accuracy : {}".format(test_accuracy / len(test_loader) * 100))
    if test_accuracy / len(test_loader) * 100 > best_accuracy:
        best_accuracy = test_accuracy / len(test_loader) * 100
        torch.save(model.state_dict(), "./final.pth")
        print("Save model")
    print("=================================================")


# %%


# Plot loss and accuracy
fig = plt.figure(figsize=(15, 6))  
sub1 = fig.add_subplot(1, 2, 1) 
sub2 = fig.add_subplot(1, 2, 2) 


sub1.set_xlabel("Epochs")
sub1.set_ylabel("Accuracy %")

sub1.plot(train_accuracy_history, color="green", label="Training ACC")
sub1.plot(test_accuracy_history, color="orange", label="Test ACC")
sub1.legend(loc=4)

sub2.set_xlabel("Epochs")
sub2.set_ylabel("Loss")

sub2.plot(total_train_loss_history, color="green", label="Training Loss")
sub2.plot(total_test_loss_history, color="orange", label="Test Loss")
sub2.legend(loc=1)

plt.savefig('swin_base_patch4_window12_384_in22k.png')


# %%

# Load weight with best accuracy
# model.load_state_dict(torch.load("./final.pth"))


# %%


# fixed testing process
# correct = 0
# total = 0
# # since we're not training, we don't need to calculate the gradients for our outputs
# with torch.no_grad():
#     for data in test_loader:
#         images, labels = data
#         images = images.to(device)
#         labels = labels.to(device)
#         # calculate outputs by running images through the network
#         outputs = model(images)
#         # the class with the highest energy is what we choose as prediction
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
# print(f'Accuracy of the network on the 10000 test images: {100 * correct / total:.2f} %')

