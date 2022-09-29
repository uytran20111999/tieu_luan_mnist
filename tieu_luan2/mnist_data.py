from random import gauss
from torchvision.datasets import MNIST
import pytorch_lightning as pl
from torchvision import transforms
from torch.utils.data import random_split, DataLoader
import torch
import numpy as np

class RandomGaussNoise(object):
    def __init__(self,mean,std,p=0.5):
        self.mean = mean
        self.std = std
        self.prop = p
    
    def __call__(self, x):
        choice = np.random.choice([0, 1], p=[1-self.prop, self.prop])
        noise = self.std*torch.randn(x.shape)+self.mean
        res = noise + x if choice else x
        return res.clamp_(0,1)


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self,batch_size =32,data_dir:str = './data/',**kwargs) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.train_transform = transforms.Compose([transforms.ToTensor(),
                                            #transforms.RandomHorizontalFlip(0.5),
                                            transforms.RandomApply([transforms.RandomAffine(0,translate = [0.2,0.2])],0.5),
                                            transforms.RandomApply([transforms.RandomAffine((-20,20))],0.5),
                                            RandomGaussNoise(0,0.05,0.5),
                                             transforms.Normalize((0.1307,), (0.3081,))])

        self.test_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
        self.batch_size = batch_size
        self.save_hyperparameters()

    def prepare_data(self):
        MNIST(self.data_dir,train=True,download=True)
        MNIST(self.data_dir,train=False,download=True)

    def setup(self,stage: str):
        if stage == 'fit':
            mnist_full = MNIST(self.data_dir,train=True,transform=self.train_transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full,[48000,12000])
        if stage == 'test':
            self.mnist_test = MNIST(self.data_dir,train = False,transform=self.test_transform)
        if stage == 'predict':
            self.mnist_test = MNIST(self.data_dir,train = False,transform=self.test_transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size,shuffle=True,num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size,num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size,num_workers=4)

    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, batch_size=self.batch_size,num_workers=4)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("mnist_ds")
        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument("--data_dir", type=str, default='./data/')
        return parent_parser


if __name__ == '__main__':
    mnist_dm = MNISTDataModule()
    mnist_dm.prepare_data()
    mnist_dm.setup('fit')
    train_ds = MNIST('./data/',train=True)
    trains_tf = mnist_dm.train_transform
    # train_dl = mnist_dm.train_dataloader()
    # get_img = next(iter(train_dl))
    # imgs, lbs = get_img

    # imgs = imgs[0]
    # lbs = lbs[0]

    # img,lb = train_ds[5]
    # aug_img = trains_tf(img)

    # import matplotlib.pyplot as plt
    # fig,ax = plt.subplots(1,2)
    # original_ax = ax[0]
    # augment_ax = ax[1]

    # ccc = img
    
    # original_ax.imshow(ccc,cmap='gray')
    # original_ax.set_title(f'Ảnh gốc: nhãn {lb}')

    # augment_ax.imshow(aug_img.squeeze(),cmap='gray')
    # augment_ax.set_title(f'Ảnh kết hợp: nhãn {lb}')
    # plt.show()
    import matplotlib.pyplot as plt
    fig,ax = plt.subplots(3,3)
    indices = torch.randperm(len(train_ds))[:9]
    for i in range(3):
        for j in range(3):
            pic_index = i*3+j
            img,lb = train_ds[indices[pic_index]]
            ax[i][j].imshow(img,cmap='gray')
            ax[i][j].set_title(f'Nhãn {lb}')
            ax[i][j].set_xticks([])
            ax[i][j].set_yticks([])
    fig.tight_layout()
    plt.show()





    

