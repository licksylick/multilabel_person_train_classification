### <p align="center">Multi-label person/train classification</p>  

<p align="center"><img src="https://i.ibb.co/q5MxxqV/screenshot-2.png" height="300"></p>  


[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1aZujmHUjzw5CPbIVwpYB2wt1kMKXy7Xx?usp=sharing)
## 🚀 Train

### 1. Install all necessary libs:
  ```sh
  pip3 install -r requirements.txt
  ```
Note: if you are using a GPU, then you need to install CUDA and replace the torch version in `requirements.txt` with the GPU-enabled version.
Otherwise, the processor will be used.


-----
### 2. Edit `config.py` (can skip)

-----
### 3. Run the training script with the arguments:
```sh
python3 train.py --model=resnet34 --pretrained=True --epoch_num=40 --checkpoints=ckpts
  ```
 you can choose as the `model` argument:
 * ResNet18 ('resnet18')
 * ResNet34 ('resnet34')
 * ResNet50 ('resnet50')
 * Efficientnet b0-b7 ('efficientnet-b0' etc.)

-----
-----
## ✅ Inference
###  Run `inference.py`, specifying the required architecture, the path to the model and the image in the arguments:
  ```sh
  python3 inference.py --model_arch=resnet34 --ckpt=model.ckpt --image_path=image.jpg
  ```
  
