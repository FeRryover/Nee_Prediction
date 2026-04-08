import numpy as np
import copy
import torch

def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj=='type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch-1) // 1))}
    elif args.lradj=='type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6, 
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False         ###早停标志
        self.val_loss_min = np.Inf      ###这一行将实例变量val_loss_min设置为正无穷
        self.delta = delta
        self.best_state_dict = None

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        self.best_state_dict = copy.deepcopy(model.state_dict())
        self.val_loss_min = val_loss

    def load_best_weights(self, model):
        if self.best_state_dict is not None:
            model.load_state_dict(self.best_state_dict)

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class StandardScaler():
    def __init__(self):
        '''这个方法的目的是初始化类的属性'''
        self.mean = 0.
        self.std = 1.
    
    def fit(self, data):
        '''计算输入数据的均值和方差'''
        self.mean = data.mean(0)
        self.std = data.std(0)

    def transform(self, data):
        '''
        通过torch.from_numpy方法将mean和std属性转换为PyTorch的Tensor类型。然后，使用type_as方法将它们转换为与输入数据相同的数据类型，
        并使用to方法将它们移动到与输入数据相同的设备上。这样做的目的是确保输入数据和mean、std的类型和设备匹配，以便进行数值计算。
        如果输入数据是一个PyTorch张量，则返回值也是一个张量。如果输入数据不是张量，则返回值是一个NumPy数组
        '''
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data - mean) / std

    def inverse_transform(self, data):
        '''
        数据类型转化
        函数通过将数据乘以标准差，然后加上均值来逆转标准化。最终输出原始值。
        '''
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        if data.shape[-1] != mean.shape[-1]:
            mean = mean[-1:]
            std = std[-1:]
        return (data * std) + mean