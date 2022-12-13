import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import zipfile
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split
import warnings

from dataset import PersonTrainDataModule
from utils import get_model, create_dataframe, download_by_url
from config import BATCH_SIZE, DATA_PATH, NUM_CLASSES, TEST_SIZE

warnings.filterwarnings('ignore')


def main():
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument('--model', type=str, required=True, help='Model name (resnet18|resnet34|resnet50|efficientnet')
    parser.add_argument('--pretrained', type=bool, default=False,
                        help='Bool value that means using of pretrained weights')
    parser.add_argument('--epoch_num', type=int, required=True, help='Number of epochs')
    parser.add_argument('--data_path', type=str, default=DATA_PATH, help='Path to data directory')
    parser.add_argument('--tune', type=bool, default=False, help='Tune model before training (find LR and batch size )')
    parser.add_argument('--log_dir', type=str, default='default', help='Log directory name')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size')
    parser.add_argument('--checkpoints', type=str, required=True, help='Directory name where to save checkpoints')
    args = parser.parse_args()

    if not os.path.exists('data'):
        dataset_url = "https://getfile.dokpub.com/yandex/get/https://disk.yandex.ru/d/Ypp_Tfy10MrRrg"
        filename = "data.zip"
        download_by_url(dataset_url, filename)
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall('data')
        os.remove(filename)
        print('Dataset successfully downloaded to data/\n')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = get_model(args.model, NUM_CLASSES, pretrained=args.pretrained)

    model.to(device)

    df = create_dataframe(args.data_path)

    train_df, test_df = train_test_split(df, test_size=TEST_SIZE, shuffle=True)

    print(f'Train: {len(train_df)}, Test: {len(test_df)}')

    dm = PersonTrainDataModule(train_df=train_df, test_df=test_df, batch_size=BATCH_SIZE)

    logger = TensorBoardLogger(save_dir=os.getcwd(), name='logs')
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=os.path.join(os.getcwd(), args.checkpoints),
        filename=f'{args.model}' + '-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        mode='min',
        verbose=True
    )

    early_stoping_callback = EarlyStopping('val_loss', patience=7)
    if device == 'cuda':
        trainer = pl.Trainer(accelerator='gpu', gpus=1, devices=-1, max_epochs=args.epoch_num)
    else:
        trainer = pl.Trainer(accelerator='cpu', max_epochs=args.epoch_num)
    trainer.logger = logger
    trainer.callbacks = [checkpoint_callback, early_stoping_callback]
    trainer.fit(model=model, datamodule=dm)

    print('Best model with loss {:.4f} located in {}'.format(checkpoint_callback.best_model_score,
                                                             checkpoint_callback.best_model_path))

    trainer.test(model=model, datamodule=dm)


if __name__ == '__main__':
    main()
