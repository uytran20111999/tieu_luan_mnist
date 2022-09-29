from argparse import ArgumentParser
from gc import callbacks
from cnn_model import lit_warper
from mnist_data import MNISTDataModule
import pytorch_lightning as pl
import torch
import numpy as np
import random
from pytorch_lightning.callbacks import ModelCheckpoint

def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def init_args():
    parser = ArgumentParser()
    parser.add_argument('--ckpt_path',type = str,default='')
    parser.add_argument('--n_epoch',type = int,default=10)
    parser = lit_warper.add_model_specific_args(parser)
    parser = MNISTDataModule.add_argparse_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    return parser

def train():
    seed_everything(10)
    parser = init_args()
    args = parser.parse_args()
    dict_args = vars(args)
    model = lit_warper(cfg = dict_args)
    mnist_data = MNISTDataModule(**dict_args)
    ckpt_cfg = ModelCheckpoint(save_top_k=3,every_n_train_steps=5,filename='mini-noaug_model-{epoch}-{val_loss:.2f}-{valid_acc_epoch:.2f}',monitor='valid_acc_epoch',mode = 'max')
    trainer = pl.Trainer.from_argparse_args(args,callbacks=[ckpt_cfg])
    trainer.fit(model,datamodule = mnist_data)

def test():
    seed_everything(10)
    parser = init_args()
    args = parser.parse_args()
    dict_args = vars(args)
    ckpt_path = dict_args['ckpt_path']
    model = lit_warper.load_from_checkpoint(ckpt_path)
    mnist_data = MNISTDataModule(**dict_args)
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.test(model,datamodule = mnist_data)
    


if __name__ =='__main__':
    test()
    #pass
    #train()