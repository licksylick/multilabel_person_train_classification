### <p align="center">Multi-label person/train classification</p>  

<p align="center"><img src="https://i.ibb.co/q5MxxqV/screenshot-2.png" height="300"></p>  


[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1aZujmHUjzw5CPbIVwpYB2wt1kMKXy7Xx?usp=sharing)
## 🚀 Тренировка модели

### 1. Установка необходимых библиотек:
  ```sh
  pip3 install -r requirements.txt
  ```
Важно! Если Вы используете GPU, то необходимо установить CUDA и заменить в `requirements.txt` версию torch на версию с поддержкой GPU. 
В противном случае будет задействован процессор.


-----
### 2. Установка переменных в `config.py` (можно пропустить)

-----
### 3. Запуск скрипта тренировки:
```sh
python3 train.py --model=resnet34 --pretrained=True --epoch_num=40 --checkpoints=ckpts
  ```
 В качестве аргумента `model` могут быть выбраны:
 * ResNet18 ('resnet18')
 * ResNet34 ('resnet34')
 * ResNet50 ('resnet50')
 * Efficientnet b0-b7 ('efficientnet-b0' etc.)

-----
-----
## ✅ Инференс
###  Необходимо запустить `inference.py`, указав в аргументах путь к модели и изображению:
  ```sh
  python3 inference.py --model_arch=resnet34 --ckpt=model.ckpt --image_path=image.jpg
  ```
  
