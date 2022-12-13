import torch
import cv2
from argparse import ArgumentParser

from utils import get_model, inference
from config import NUM_CLASSES

parser = ArgumentParser()

parser.add_argument('--model_arch', type=str, required=True, help='Model architecture (resnet34, efficientnet-b2 etc.)')
parser.add_argument('--ckpt', type=str, required=True, help='Path to model checkpoint')
parser.add_argument('--image_path', type=str, required=True, help='Path to input image')
args = parser.parse_args()


image = cv2.imread(args.image_path)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
checkpoint = torch.load(args.ckpt)
model = get_model(args.model_arch, NUM_CLASSES, False)
model.load_state_dict(checkpoint['state_dict'])
model.to(device)
model.eval()

output, text_prediction = inference(model, image)
print(f'Model output: {output}')
print(f'Predicted objects on image: {text_prediction}')
