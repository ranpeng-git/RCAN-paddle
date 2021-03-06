import os
import math
import time
import datetime
from functools import reduce

import imageio
from PIL import Image #PIL pakage name is Pillow

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import scipy.misc as misc

import paddle
import paddle.optimizer as optim
import paddle.optimizer.lr as lrs

class timer():
    def __init__(self):
        self.acc = 0
        self.tic()

    def tic(self):
        self.t0 = time.time()

    def toc(self):
        return time.time() - self.t0

    def hold(self):
        self.acc += self.toc()

    def release(self):
        ret = self.acc
        self.acc = 0

        return ret

    def reset(self):
        self.acc = 0

class checkpoint():
    def __init__(self, args):
        self.args = args
        self.ok = True
        self.log = paddle.to_tensor(np.zeros([1 , 1]) , dtype = 'float32')
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

        if args.load == '.':
            if args.save == '.': args.save = now
            self.dir = '../experiment/' + args.save
        else:
            self.dir = '../experiment/' + args.load
            if not os.path.exists(self.dir):
                args.load = '.'
            else:
                self.log = paddle.load(self.dir + '/psnr_log.pdparams')
                print('Continue from epoch {}...'.format(len(self.log)))

        if args.reset:
            os.system('rm -rf ' + self.dir)
            args.load = '.'

        def _make_dir(path):
            if not os.path.exists(path): os.makedirs(path)

        _make_dir(self.dir)
        _make_dir(self.dir + '/model')
        _make_dir(self.dir + '/results')

        open_type = 'a' if os.path.exists(self.dir + '/log.txt') else 'w'
        self.log_file = open(self.dir + '/log.txt', open_type)
        with open(self.dir + '/config.txt', open_type) as f:
            f.write(now + '\n\n')
            for arg in vars(args):
                f.write('{}: {}\n'.format(arg, getattr(args, arg)))
            f.write('\n')

    def save(self, trainer, epoch, is_best=False):
        trainer.model.save(self.dir, epoch, is_best=is_best)
        trainer.loss.save(self.dir)
        trainer.loss.plot_loss(self.dir, epoch)

        self.plot_psnr(epoch)
        paddle.save(self.log, os.path.join(self.dir, 'psnr_log.pdparams'))
        paddle.save(
            trainer.optimizer.state_dict(),
            os.path.join(self.dir, 'optimizer.pdparams')
        )

    def add_log(self, log):
        self.log = paddle.concat([self.log, log])

    def write_log(self, log, refresh=False):
        print(log)
        self.log_file.write(log + '\n')
        if refresh:
            self.log_file.close()
            self.log_file = open(self.dir + '/log.txt', 'a')

    def done(self):
        self.log_file.close()

    def plot_psnr(self, epoch):
        axis = np.linspace(1, epoch, epoch)
        label = 'SR on {}'.format(self.args.data_test)
        fig = plt.figure()
        plt.title(label)
        for idx_scale, scale in enumerate(self.args.scale):
            plt.plot(
                axis,
                self.log[:, idx_scale].numpy(),
                label='Scale {}'.format(scale)
            )
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('PSNR')
        plt.grid(True)
        plt.savefig('{}/test_{}.pdf'.format(self.dir, self.args.data_test))
        plt.close(fig)

    def save_results(self, filename, save_list, scale):
        filename = '{}/results/{}_x{}_'.format(self.dir, filename, scale)
        postfix = ('SR', 'LR', 'HR')
        for v, p in zip(save_list, postfix):
            normalized = v[0].detach()*(255 / self.args.rgb_range)
            ndarr = paddle.cast(normalized.transpose(perm =[1, 2, 0]), 'uint8').cpu().numpy()
            #misc.imsave('{}{}.png'.format(filename, p), ndarr)
            imageio.imwrite('{}{}.png'.format(filename, p), ndarr)

def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return (img*(pixel_range)).clip(0,255).round() /(pixel_range)

def calc_psnr(sr, hr, scale, rgb_range, benchmark=False):
    diff = ((sr - hr).detach())/rgb_range
    shave = scale
    if diff.shape[1]> 1:
        # convert = diff.new(1, 3, 1, 1)
        convert = paddle.zeros([1, 3, 1, 1])
        convert[0, 0, 0, 0] = 65.738
        convert[0, 1, 0, 0] = 129.057
        convert[0, 2, 0, 0] = 25.064
        diff=(diff*convert)/256
        diff = diff.sum(axis=1, keepdim=True)
    '''
    if benchmark:
        shave = scale
        if diff.size(1) > 1:
            convert = diff.new(1, 3, 1, 1)
            convert[0, 0, 0, 0] = 65.738
            convert[0, 1, 0, 0] = 129.057
            convert[0, 2, 0, 0] = 25.064
            diff.mul_(convert).div_(256)
            diff = diff.sum(dim=1, keepdim=True)
    else:
        shave = scale + 6
    '''
    valid = diff[:, :, shave:-shave, shave:-shave]
    mse = valid.pow(2).mean()

    return -10 * math.log10(mse)

def make_optimizer(args, scheduler , my_model):
    # trainable = filter(lambda x: x.requires_grad,  my_model.parameters())
    kwargs ={}
    kwargs['weight_decay'] = args.weight_decay
    if args.optimizer == 'SGD':
        kwargs = {'momentum': args.momentum}
        optimizer_function = optim.Momentum(
            learning_rate = scheduler,
            parameters = my_model.parameters(),
            **kwargs
        )
        
    elif args.optimizer == 'ADAM':
        kwargs = {
            'beta1' :args.beta1,
            'beta2': args.beta2,
            'epsilon': args.epsilon
        }
        optimizer_function = optim.Adam(
            learning_rate = scheduler,
            parameters = my_model.parameters(),
            **kwargs
        )
        
    elif args.optimizer == 'RMSprop':
        kwargs = {'epsilon': args.epsilon}
        optimizer_function = optim.RMSprop(
            learning_rate = scheduler,
            parameters = my_model.parameters(),
            **kwargs
        )
        
    # kwargs['learning_rate'] = scheduler
    
    return optimizer_function

def make_scheduler(args):
    if args.decay_type == 'step':
        scheduler = lrs.StepDecay(
            args.lr,
            step_size=args.lr_decay,
            gamma=args.gamma
        )
    elif args.decay_type.find('step') >= 0:
        milestones = args.decay_type.split('_')
        milestones.pop(0)
        milestones = list(map(lambda x: int(x), milestones))
        scheduler = lrs.MultiStepDecay(
            args.lr,
            milestones=milestones,
            gamma=args.gamma
        )

    return scheduler

