import torch
import numpy as np
import cv2
import math
from matplotlib import pyplot as plt
from torch.utils import data
import lpips

def visualization(feature_maps):
    nums = feature_maps.size(1)
    nrows = int(math.sqrt(nums))
    ncols = nums // nrows + 1
    plt.figure()
    for i in range(nums):
        img = feature_maps[:, i].squeeze().detach().cpu()
        plt.subplot(nrows, ncols, i+1)
        plt.imshow(img, interpolation='bicubic')
    plt.show()

class MyDataset(data.Dataset):
    def __init__(self, path, len_input, interval=1):
        self.data = torch.load(path)
        t, b, c, h, w = self.data.size()
        self.data = self.data.reshape(t * b, c, h, w)
        self.len_seq = self.data.size(0)
        self.interval = interval
        self.len_input = len_input
        self.nodes = []

    def __len__(self):
        len_index = 0
        for i in range(self.interval):
            len_index += ((self.len_seq + (self.interval - 1) - i) // (self.len_input * self.interval))
            self.nodes.append(len_index)
        return len_index

    def __getitem__(self, item):
        for round in range(len(self.nodes)):
            if item < self.nodes[round]:
                if round - 1 >= 0:
                    item = item - self.nodes[round-1]
                break
        input_seq = [self.data[i:i+1, :] for i in range(round + item * self.len_input * self.interval, round + (item + 1) * self.len_input *self.interval, self.interval)]
        return torch.cat(input_seq, dim=0)


class util_of_lpips():
    def __init__(self, net):
        self.loss_fn = lpips.LPIPS(net=net)


    def calc_lpips(self, predict, target):
        device = predict.device
        self.loss_fn.to(device)
        dist01 = self.loss_fn.forward(predict, target, normalize=True)
        return dist01


def ShowImages(predictions, targets):
    pre_imgs = []
    tar_imgs = []

    for i in range(0, len(predictions), 2):
        pre = predictions[i]
        pre = pre.cpu().numpy().squeeze()
        pre = np.transpose(pre, (1, 2, 0))
        pre_imgs.append(pre)
        target = targets[:, i]
        target = target.cpu().squeeze().numpy()
        target = np.transpose(target, (1, 2, 0))
        tar_imgs.append(target)

    stack_img = np.hstack(pre_imgs)
    stack_tar = np.hstack(tar_imgs)
    cv2.imshow('stack_img', stack_img)
    cv2.imshow('tar_img', stack_tar)

    d = cv2.waitKey()
    print('press "Esc" to directly destroy all windows, or other keys to save current images')
    if d == 27:
        cv2.destroyAllWindows()
    else:
        cv2.destroyAllWindows()
        save_img = input('Save image? Type "y" to save the image, otherwise skip')
        if save_img == 'y':
            name = input('Type save name: ')
            save_name = './predictions/KTH/long_term/{}_pred.png'.format(name)
            cv2.imwrite(save_name, stack_img*255)
            tar_name = './predictions/KTH/long_term/{}_real.png'.format(name)
            cv2.imwrite(tar_name, np.hstack(tar_imgs))
