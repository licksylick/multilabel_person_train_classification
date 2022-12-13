import torch
import pytorch_lightning as pl
import torchvision.models as models
import torch.nn.functional as F
from torch import nn
from torchmetrics.functional.classification import multilabel_f1_score, multilabel_precision, multilabel_recall


class Resnet(pl.LightningModule):

    def __init__(self, model_name: str, num_classes: int, pretrained=False):
        super().__init__()
        if model_name == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
        elif model_name == 'resnet34':
            self.backbone = models.resnet34(pretrained=pretrained)
        elif model_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
        else:
            raise ValueError(f'Undefined value of model name: {model_name}')

        self.num_classes = num_classes
        in_features = self.backbone.fc.in_features

        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.Linear(1024, num_classes),
            nn.Sigmoid()
        )


    def forward(self, x: torch.Tensor):
        return self.backbone(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return [optimizer]

    def compute_metrics(self, pred, target):
        metrics = dict()
        metrics['f1'] = multilabel_f1_score(pred, target, num_labels=self.num_classes)
        metrics['precision'] = multilabel_precision(pred, target, num_labels=self.num_classes)
        metrics['recall'] = multilabel_recall(pred, target, num_labels=self.num_classes)
        return metrics

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        loss = F.binary_cross_entropy(output, y)
        metrics = self.compute_metrics(output, y)
        self.log('train_f1', metrics['f1'], prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_prec', metrics['precision'], prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_rec', metrics['recall'], prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        loss = F.binary_cross_entropy(output, y)
        metrics = self.compute_metrics(output, y)
        self.log('val_f1', metrics['f1'], on_step=False, on_epoch=True)
        self.log('val_prec', metrics['precision'], on_step=False, on_epoch=True)
        self.log('val_rec', metrics['recall'], on_step=False, on_epoch=True)
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        loss = F.binary_cross_entropy(output, y)
        metrics = self.compute_metrics(output, y)
        self.log('test_f1', metrics['f1'], on_step=False, on_epoch=True)
        self.log('test_prec', metrics['precision'], on_step=False, on_epoch=True)
        self.log('test_rec', metrics['recall'], on_step=False, on_epoch=True)
        self.log('test_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss
