import numpy as np
import torch
import matplotlib.pyplot as plt



def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        if(epoch<=10):
            lr_adjust = {epoch: args.learning_rate * (0.5 ** (((epoch - 1) // 2)))}
        else:
            lr_adjust = {
                10: 1e-5, 15: 5e-6, 30: 5e-7, 50: 1e-8, 75: 1.5e-8, 90: 1e-10
            }
    elif args.lradj == 'type2':
        # lr_adjust = {
        #     2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
        #     10: 5e-7, 15: 1e-7, 20: 5e-8, 30: 1e-8, 40: 5e-9, 50: 1e-9, 75: 5e-10, 90: 1e-10
        # }
        lr_adjust = {
         1: 1e-3, 2: 5e-4, 4:2.5e-4, 6:1e-4, 8: 5e-5,
         10: 1e-5, 15: 5e-6, 20: 1e-6, 30: 5e-7, 40: 1e-7, 50: 1e-8, 75: 1e-9, 90: 1e-10
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
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.test_loss_min = np.Inf
        self.delta = delta

    def __call__(self, test_loss, model, path):
        score = -test_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(test_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(test_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, test_loss, model, path):
        if self.verbose:
            print(f'Validation(test) loss decreased ({self.test_loss_min:.6f} --> {test_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.test_loss_min = test_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')
