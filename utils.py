import os
import torch
import pandas as pd
import requests
from tqdm import tqdm
import warnings

from models.resnet import Resnet
from models.efficientnet import EfficientNetModel
from augmentations import test_transforms
from config import THRESHOLD

warnings.filterwarnings("ignore")


def download_by_url(url: str, fname: str):
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get('content-length', 0))
    with open(fname, 'wb') as file, tqdm(
        desc=fname,
        total=total,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)


def get_model(model_name, num_classes, pretrained):
    if 'resnet' in model_name:
        model = Resnet(model_name, num_classes, pretrained)
    elif 'efficientnet' in model_name:
        model = EfficientNetModel(model_name, num_classes)
    else:
        raise ValueError(f'Undefined model name: {model_name}')
    return model


def create_dataframe(data_path):
    df = pd.DataFrame(columns=['image', 'person', 'train'])
    for path, subdirs, files in os.walk(data_path):
        class_dir = path.split('/')[-1]
        for name in files:
            if '.DS_Store' not in name:
                if str(class_dir) == 'person_train':
                    df = df.append({'image': os.path.join(path, name),
                                    'person': 1, 'train': 1}, ignore_index=True)
                elif str(class_dir) == 'person':
                    df = df.append({'image': os.path.join(path, name),
                                    'person': 1, 'train': 0}, ignore_index=True)
                elif str(class_dir) == 'train':
                    df = df.append({'image': os.path.join(path, name),
                                    'person': 0, 'train': 1}, ignore_index=True)
                else:
                    df = df.append({'image': os.path.join(path, name),
                                    'person': 0, 'train': 0}, ignore_index=True)
    return df


def inference(model, image, threshold=THRESHOLD):
    image = test_transforms(image).unsqueeze_(0)

    prediction = model(image)

    pred_arr = prediction.detach().numpy()[0]
    indices = [i for i, v in enumerate(pred_arr >= threshold) if v]

    classes = ['person(s) ', 'train(s) ', 'no person(s), no train(s) ']
    res = ''
    if len(indices) > 0:
        for i in indices:
            res += classes[i]
    else:
        res = 'no person(s), no train(s)'
    return pred_arr, res
