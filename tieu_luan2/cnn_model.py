from typing import Tuple
import pytorch_lightning as pl
import torch
import config
from torchmetrics import Accuracy,Recall,Precision,F1Score

# 2 resblocks

# 28x28x1 -> 14x14x16 -> 7x7x32 -> 1x1x32 -> linear


class cnn_blocks(torch.nn.Module):
    def __init__(self,in_channels:int
                ,out_channels:int,
                use_batchnorm : bool, 
                activations_funcname:str,**kwargs) -> None:
        '''
        Simple Convolution Block which comprises 3 basic components (Conv, Batch_Norm [optional], activations_name [optionals])
        args:
        in_channels: input dim
        out_channels: output dim
        use_batchnorm: bool variable which indicates if bn is used
        activations_funcname: string name of the desire activation
        '''
        super().__init__()
        act_fn = {'relu':torch.nn.ReLU(),
                'p_relu':torch.nn.PReLU(num_parameters=out_channels),
                'hard_swish':torch.nn.Hardswish()}

        assert activations_funcname is None or activations_funcname in act_fn, f'activations name must be none or must be one of {list(act_fn.keys())}'
        self.conv = torch.nn.Conv2d(in_channels=in_channels,out_channels=out_channels,bias = not use_batchnorm,**kwargs)
        self.act = act_fn.get(activations_funcname,torch.nn.Identity())
        self.bn = [torch.nn.Identity(),torch.nn.BatchNorm2d(num_features=out_channels)][use_batchnorm]
    
    def forward(self,x:torch.Tensor):
        '''
        x: input tensor
        '''
        return self.act(self.bn(self.conv(x)))


class residual_blocks(torch.nn.Module):
    def __init__(self,in_channels,out_channels,nums_convs,use_res,use_bn) -> None:
        super().__init__()
        self.model = []
        for i in range(nums_convs-1):
            self.model.append(cnn_blocks(in_channels,in_channels,False,'p_relu',kernel_size = (3,3),padding = 1))  
        self.model.append(cnn_blocks(in_channels,out_channels,use_bn,'p_relu',kernel_size = (3,3),padding = 1))
        self.model = torch.nn.Sequential(*self.model)
        self.use_res = use_res
        self.res = None
        if self.use_res:
            self.res = torch.nn.Identity() if in_channels == out_channels else cnn_blocks(in_channels,out_channels,False,None,kernel_size = (1,1))
    
    def forward(self,x):
        res = self.model(x)
        return res if not self.use_res else res+self.res(x)


class MyModel(torch.nn.Module):
    def __init__(self,arc) -> None:
        '''
        arc:
        is a list of tuples
        whose format is specified in the config.py.
        '''
        super().__init__()
        self.arc = arc
        self.model = self._build_model()
    
    def _build_model(self):
        layers = []
        for layer in self.arc:
            if layer[0] == 'R':
                num_cv, in_channels, out_channel = layer[1:]
                layers.append(residual_blocks(in_channels,out_channel,num_cv,True,True))
            elif layer[0] =='D':
                in_channels,out_channels, activation_name = layer[1:]
                layers.append(cnn_blocks(in_channels,out_channels,False,activation_name,kernel_size = 3,stride = 2,padding = 1))
            elif layer[0]=='G':
                layers.append(torch.nn.AdaptiveAvgPool2d((1,1)))
            elif layer[0]=='L':
                in_channels,out_channels, activation_name = layer[1:]
                layers.append(cnn_blocks(in_channels,out_channels,False,activation_name,kernel_size = 1))
                self.num_class = out_channels
        return torch.nn.Sequential(*layers)

    def forward(self,x):
        B,_,_,_ = x.shape
        return self.model(x).reshape(B,self.num_class)


class lit_warper(pl.LightningModule):
    def __init__(self, cfg ,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters('cfg')
        self.cfg = cfg
        self.model = MyModel(config.model_config[self.cfg['arc']])
        self.loss = config.get_loss_func[self.cfg['loss']]
        self.trainer_valid_keys = ['lr','momentum','weight_decay']
        self.accuracy = Accuracy()
        self.valid_acc = Accuracy()
        self.test_acc = Accuracy()

    def configure_optimizers(self):
        optim_name= self.cfg['optim']
        valid_arg = {i:self.cfg[i] for i in self.trainer_valid_keys if self.cfg[i] is not None}
        return config.valid_optim[optim_name](self.model.parameters(),**valid_arg)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        preds = self.model(x)
        accuracy = self.accuracy(preds.argmax(dim = -1).squeeze(),y.squeeze())
        loss = self.loss(preds, y)

        self.log('train_loss',loss)
        self.log('train_acc_step',self.accuracy,prog_bar=True)
        return {'loss':loss,'acc':accuracy}

    def training_epoch_end(self, outs):
        # log epoch metric
        self.log('train_acc_epoch', self.accuracy,prog_bar=True)

    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, y = batch
        preds = self.model(x)
        test_acc = self.test_acc(preds.argmax(dim = -1).squeeze(),y.squeeze())
        test_loss = self.loss(preds, y)
        self.log("test_loss", test_loss)
        self.log('test_acc',self.test_acc,prog_bar=True)
        return test_loss

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, y = batch
        preds = self.model(x)
        self.valid_acc(preds.argmax(dim = -1).squeeze(),y.squeeze())
        val_loss = self.loss(preds, y)
        self.log("val_loss", val_loss)
        self.log('valid_acc_step',self.valid_acc,prog_bar=True)
        return val_loss

    def validation_epoch_end(self, outputs) -> None:
        self.log('valid_acc_epoch', self.valid_acc,prog_bar=True)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("simple_cnn")
        parser.add_argument("--lr", type=float, default=0.01)
        parser.add_argument('--momentum',type=float,default = None)
        parser.add_argument('--weight_decay',type=float,default = None)
        parser.add_argument('--arc',type=str,default = 'default_model')
        parser.add_argument('--loss',type = str,default = 'cross_entropy')
        parser.add_argument('--optim',type = str,default = 'SGD')
        return parent_parser


if __name__ == '__main__':
    #x = torch.randn([16,1,28,28])
    simple_cnn = MyModel(config.model_config)
    for i in range(10):
        x = torch.randn([16,1,28,28])
        res = simple_cnn(x)
        print(res.shape)



