import pytorch_lightning as pl
import torch


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
        self.conv = torch.nn.Conv2d(in_channels=in_channels,out_channels=out_channels,**kwargs)



class residual_blocks(torch.nn.Module):
    pass

class se_blocks(torch.nn.Module):
    pass

class sa_blocks(torch.nn.Module):
    pass


class MyModel(torch.nn.Module):
    def __init__(self,nums_layers,nums_class) -> None:
        super().__init__()

