from dataclasses import dataclass
from turtle import forward
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch.utils.data import Dataset,DataLoader
from tqdm import tqdm
import time
import random
import os

def true_cosine_function(x:torch.Tensor):
    '''
    x in [0,2pi]
    '''
    return torch.sin(x)

def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def gen_fake_data(functions,nums_data=2,max_sampled_data = 10000):
    if nums_data >= max_sampled_data:
        nums_data = max_sampled_data
    sampled_data = torch.arange(0,2*torch.pi,2*torch.pi/max_sampled_data)
    indices = torch.LongTensor(np.random.choice(max_sampled_data,nums_data, replace=False)) 
    x = torch.sort(sampled_data[indices])[0]
    return x , functions(x)


def plot_data_and_ponts(ax , true_data_x,true_data_y,sampled_data_x,sampled_data_y,is_plot_points = True):
    ax.plot(true_data_x, true_data_y,'r--',label = 'sin(x)')
    #plt.show()
    if is_plot_points:
        ax.set_title('Dữ liệu quan sát được từ hàm sin(x)')
        ax.scatter(sampled_data_x,sampled_data_y,s=12,c = 'b',label = 'dữ liệu quan sát được')
    else:
        ax.set_title('sin(x) và hàm được học')
        ax.plot(sampled_data_x,sampled_data_y,'b',label = 'hàm số được học')
    ax.set_xlabel('X')
    ax.set_xlabel('Y')
    ax.legend()


class poly_model(torch.nn.Module):
    def __init__(self,num_dim = 3) -> None:
        super().__init__()
        self.mult = torch.nn.parameter.Parameter(torch.randn(num_dim,1))
        self.bias = torch.nn.parameter.Parameter(torch.zeros(1))
        self.proj = torch.nn.parameter.Parameter(torch.randn(1))
        self.proj_bias = torch.nn.parameter.Parameter(torch.zeros(1))
        self.num_dim = num_dim
        self.tanh = torch.nn.Tanh()
        #self.p_relu = torch.nn.Hardswish(inplace=True)
    def forward(self,x):
        #Bx 1
        x = self.tanh(self.proj*x+self.proj_bias)
        trans_formed = []
        for i in range(self.num_dim):
            trans_formed.append(x**(i+1))
        #Bx num_dim
        tmp = torch.cat(trans_formed,-1)
        return tmp@self.mult+self.bias

   
class SamplePointsData(Dataset):
    def __init__(self,points_x: torch.Tensor,points_y:torch.Tensor) -> None:
        super().__init__()
        assert points_x.shape[0]==points_y.shape[0]
        self.x = points_x
        self.y = points_y

    def __len__(self) -> int:
        return len(self.x)
    
    def __getitem__(self, index):
        return (self.x[index].unsqueeze(0),self.y[index])


class Train_Loop(object):
    
    def __init__(self, train_datasets,test_datasets,model,loss_ft,n_epochs:int=3,device = 'cuda:0',**kwargs) -> None:
        self.train_dl = DataLoader(train_datasets,shuffle=True,**kwargs)
        self.test_dl = DataLoader(test_datasets,shuffle=False,**kwargs)
        self.model = model
        self.loss_f = loss_ft
        self.optim = torch.optim.SGD(self.model.parameters(),lr=0.001,momentum=0.9)
        self.n_epochs = n_epochs
        self.model = self.model.to(device = device)
        self.device = device

    def train_one_batch(self,x,y):
        preds = self.model(x)
        loss = self.loss_f(preds.squeeze(),y.squeeze())
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return loss.detach().item()


    def test_one_batch(self,x,y):
        with torch.no_grad():
            preds = self.model(x)
            loss = self.loss_f(preds.squeeze(),y.squeeze())
            return loss.detach().item()

    def train(self):
        self.model = self.model.train()
        for epoch in range(self.n_epochs):
            train_loss = 0
            num_data = 0
            loop = tqdm(self.train_dl,leave=False,total=len(self.train_dl))
            for x,y in loop:
                x,y = x.to(device = self.device),y.to(device = self.device)
                cur_loss=self.train_one_batch(x,y)
                num_data+=x.shape[0]
                loop.set_description(f'Epoch [{epoch}/{self.n_epochs}]')
                loop.set_postfix(loss = cur_loss,avg_loss = train_loss/num_data)
                train_loss+=cur_loss
            train_loss/=num_data
            

    def test(self):
        self.model = self.model.eval()
        loop = tqdm(self.test_dl,leave=False,total=len(self.test_dl))
        test_loss = 0
        num_data = 0
        for x,y in loop:
            x,y = x.to(device = self.device),y.to(device = self.device)
            cur_loss=self.test_one_batch(x,y)
            num_data+=x.shape[0]
            loop.set_postfix(loss = cur_loss,avg_loss = test_loss/num_data)
            test_loss+=cur_loss
        #print(test_loss/num_data)
        return test_loss/num_data


    def predict_one_points(self,x):
        self.model = self.model.eval()
        x = x.unsqueeze(-1)
        x = x.to(device = self.device)
        with torch.no_grad():
            return self.model(x).squeeze().detach().cpu()

def main_run(range_experiments,poly_order = 20):
    batch_size = 16
    max_sampled_data=10000
    true_x,true_y = gen_fake_data(true_cosine_function,max_sampled_data,max_sampled_data)
    test_ds = SamplePointsData(true_x,true_y)
    n_epochs = 40
    root = './tieu_luan_material1'
    nums_data_pnt = list(range_experiments)
    test_rs = []
    for i in range_experiments:
        model = poly_model(poly_order)
        num_fake_data = i
        observed_x, observed_y = gen_fake_data(true_cosine_function,num_fake_data,max_sampled_data)
        train_ds = SamplePointsData(observed_x,observed_y)
        train_loop = Train_Loop(train_ds,test_ds,model,torch.nn.L1Loss(),n_epochs,batch_size=batch_size)
        print(f'Training the model for {i} datapoints')
        train_loop.train()
        test_result = train_loop.test()
        print(f'test of {i}-datapoints-model: {test_result}')
        fig,ax = plt.subplots(2,1)
        plot_data_and_ponts(ax[0],true_x,true_y,observed_x,observed_y,True)

        mesh_x,mesh_y = gen_fake_data(train_loop.predict_one_points,max_sampled_data,max_sampled_data)
        plot_data_and_ponts(ax[1],true_x,true_y,mesh_x,mesh_y,False)
        file_name = f'numdp_{i}_order_{poly_order}.png'
        fig.tight_layout()
        plt.savefig(os.path.join(root,file_name))
        fig.clear()
        test_rs.append(test_result)

    fig,ax = plt.subplots(1,1)
    ax.plot(nums_data_pnt,test_rs)
    ax.set_title('Mối quan hệ giữa số lượng dữ liệu quan sát và độ lỗi xấp xỉ của hàm học máy')
    ax.set_xlabel('số lượng dữ liệu quan sát')
    ax.set_ylabel('độ lỗi')
    fig.tight_layout()
    plt.savefig(os.path.join(root,'aggregate_result.png'))
    







    pass

if __name__ == '__main__':
    #print(gen_fake_data(true_cosine_function))
    # fig,ax = plt.subplots(1,1)
    # max_sampled_data = 10000
    # num_fake_data = 50
    # true_x,true_y = gen_fake_data(true_cosine_function,max_sampled_data,max_sampled_data)
    # observed_x, observed_y = gen_fake_data(true_cosine_function,num_fake_data,max_sampled_data)
    # plot_data_and_ponts(ax,true_x,true_y,observed_x,observed_y,True)
    # plt.show()
    seed_everything(1)
    main_run((30,60,100,200,400,800,2000,5000))






    
