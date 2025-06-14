# model.py
import os
import torch
import torch.nn as nn
from torchvision import models
import zipfile

def unzip_model(zip_path):
    out_path = zip_path.replace(".zip", ".pt")
    if not os.path.exists(out_path):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(os.path.dirname(zip_path))
    return out_path

def get_model(name, weight_path, is_zipped=True, num_classes=5):
    if is_zipped:
        weight_path = unzip_model(weight_path)

    if name == "resnet34":
        model = models.resnet34(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError("Unsupported model name")

    # Load state_dict
    state_dict = torch.load(weight_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


