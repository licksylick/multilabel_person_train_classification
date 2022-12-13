import torch
import pytorch_lightning as pl
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from PIL import Image
from typing import Optional

from augmentations import train_transforms, test_transforms
from config import IMAGE_SIZE, BATCH_SIZE, VAL_SIZE, SEED, NUM_WORKERS


class PersonTrainDataset(Dataset):

    def __init__(self, df, train=True, train_transform=train_transforms, test_transform=test_transforms):
        super().__init__()

        self.df = df

        if train:
            self.transform = train_transform
        else:
            self.transform = test_transform

        self.data = list()
        self.targets = list()
        for path in self.df[['image']].values:
            for img_path in path:
                self.data.append(img_path)
        for label in self.df[['person', 'train']].values:
            self.targets.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        image = Image.open(self.data[idx]).convert('RGB')
        target = torch.tensor(self.targets[idx].astype('float32'))

        if self.transform:
            image = self.transform(image)

        image = image.to(device)
        target = target.to(device)

        return image, target


class PersonTrainDataModule(pl.LightningDataModule):

    def __init__(
            self, train_df, test_df, image_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE, val_size=VAL_SIZE, train_transform=train_transforms, test_transform=test_transforms,
            num_workers=NUM_WORKERS
    ):
        super().__init__()
        self.train_df = train_df
        self.test_df = test_df
        self.image_size = image_size
        self.batch_size = batch_size
        self.val_size = val_size
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None):
        self.train_data = PersonTrainDataset(df=self.train_df, train_transform=self.train_transform,
                                             test_transform=self.test_transform)
        self.test_data = PersonTrainDataset(df=self.test_df, train=False, train_transform=self.train_transform,
                                            test_transform=self.test_transform)

        indices = np.arange(len(self.train_data))
        np.random.seed(SEED)
        np.random.shuffle(indices)
        val_len = int(len(indices) * self.val_size)
        train_indices, val_indices = indices[val_len:], indices[:val_len]

        self.train_sampler = SubsetRandomSampler(train_indices)
        self.val_sampler = SubsetRandomSampler(val_indices)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_data,
            batch_size=self.batch_size,
            sampler=self.train_sampler,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.train_data,
            batch_size=self.batch_size,
            sampler=self.val_sampler,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )
