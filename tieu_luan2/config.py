import torch

model_config = {'default_model':[
    ('R',2,1,16), # Res - num_layers - in_cn - out_cn
    ('D',16,32,'p_relu'), # Downsample (conv 3x3 stride 2) - in_cn - out_cn act
    ('R',2,32,64),
    ('D',64,64,'p_relu'),
    ('G'), #global avg pool
    ('L',64,10,'p_relu') #Linear
],
'mini_model':[
    ('R',2,1,16), # Res - num_layers - in_cn - out_cn
    ('D',16,32,'p_relu'), # Downsample (conv 3x3 stride 2) - in_cn - out_cn act
    ('D',32,64,'p_relu'),
    ('G'), #global avg pool
    ('L',64,10,'p_relu') #Linear
],
}

valid_optim = {
    'Adam': torch.optim.Adam,
    'SGD':torch.optim.SGD
}

get_loss_func = {
    'cross_entropy': torch.nn.CrossEntropyLoss(label_smoothing=0.05)
}
