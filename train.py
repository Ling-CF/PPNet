import os
import numpy as np
import torch
from PPNET import PPNet
import datetime
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
import sys
import random
from utils import MyDataset, util_of_lpips
from torch.utils.data import dataloader
device = torch.device('cuda:0')


class Training():
    def __init__(self):
        self.train_path 	= './datasets/KTH/train'  # replace it with the path where you store your training dataset
        self.val_path 		= './datasets/KTH/val'	  # replace it with the path where you store your validation dataset
        self.tag 			= 'KTH.pt'                # tag for storing
        self.retrain 		= False                   # retrain or not
        self.num_epoch 		= 100                     # training epoch
        self.len_input 		= 20                      # length of the input sequence
        self.weight_factor 	= 1000                 	  # weighting factor (corresponds to the hyperparameter p)


    def val_model(self, model):
        model.eval()
        all_ssim = []
        all_psnr = []
        all_lpips = []
        lpips_loss = util_of_lpips('alex')
        with torch.no_grad():
            for curdir, dirs, files in os.walk(self.val_path):
                if len(files) == 0:
                    continue
                files.sort()
                # print(files)
                for file in files:
                    cur_path = os.path.join(curdir, file)
                    val_dataset = MyDataset(path=cur_path, len_input=10, interval=1)
                    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, num_workers=1,
                                                                 drop_last=True)
                    val_dataloader = list(val_dataloader)
                    for item in range(0, len(val_dataloader), 2):
                        # print(len(val_dataloader), list(val_dataloader))
                        if item > len(val_dataloader) - 2:
                            break
                        inputs = torch.true_divide(val_dataloader[item], 255).to(device)
                        targets = torch.true_divide(val_dataloader[item + 1], 255)
                        ssim_score = []
                        psnr_score = []
                        lpips_score = []
                        predictions = model(inputs, PredSteps=10, mode='test')
                        for t in range(10):
                            target = targets[:, t].to(device)
                            predict = predictions[t]
                            lpips = lpips_loss.calc_lpips(predict, target)
                            lpips_score.append(lpips.item())
                            target = target.squeeze().cpu().numpy()
                            target = np.transpose(target, (1, 2, 0))
                            predict = predict.data.cpu().numpy().squeeze()
                            predict = np.transpose(predict, (1, 2, 0))
                            (ssim, diff) = structural_similarity(target, predict, win_size=None, multichannel=True,
                                                                 data_range=1.0,
                                                                 full=True)
                            psnr = peak_signal_noise_ratio(target, predict, data_range=1.0)
                            psnr_score.append(psnr)
                            ssim_score.append(ssim)
                        # print(ssim_score, psnr_score, lpips_score)
                        all_ssim.append(ssim_score)
                        all_psnr.append(psnr_score)
                        all_lpips.append(lpips_score)
        all_ssim = np.array(all_ssim)
        mean_ssim = np.mean(all_ssim, axis=0)
        all_psnr = np.array(all_psnr)
        mean_psnr = np.mean(all_psnr, axis=0)
        all_lpips = np.array(all_lpips)
        mean_lpips = np.mean(all_lpips, axis=0)
        print('ssim: ', mean_ssim, '\n', 'psnr: ', mean_psnr, '\n', 'lpips: ', mean_lpips)
        return np.mean(mean_ssim), np.mean(mean_psnr), np.mean(mean_lpips)


    def StartTraining(self, rank=0):
        info = ['pix train', self.tag, 'epoch = {}'.format(self.num_epoch),
                'length of input = {}'.format(self.len_input)]
        print(info)
        if input('Start training? Enter "n" to exit') == 'n':
            sys.exit()
        print('Start training')

        state_path = './models/KTH//Model_{}'.format(self.tag)             # storage path for model state
        acc_path = './metric/KTH/Acc_{}'.format(self.tag)                  # storage path for all precision (every epoch)

        # setup model and optimizer
        model = PPNet(channels=(3, 64, 64, 256, 256, 512), weight_factor=self.weight_factor).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)

        # load model state, optimizer state, etc. if retrain
        if self.retrain:
            checkpoint = torch.load(state_path, map_location='cuda:{}'.format(0))
            model.load_state_dict(checkpoint['model_state'])
            cur_epoch = checkpoint['epoch']
            cur_loss = checkpoint['loss']
            accuracy = torch.load(acc_path)
            ssim_score = accuracy['ssim']
            psnr_score = accuracy['psnr']
            lpips_score = accuracy['lpips']
            print('Begine resume training')
        else:
            cur_epoch = 0
            cur_loss = 0
            ssim_score = []
            lpips_score = []
            psnr_score = []

        start_time = datetime.datetime.now()
        print('start time: ', start_time.strftime('%H:%M:%S'))

        assert cur_epoch < self.num_epoch

        for epoch in range(cur_epoch, self.num_epoch):
            model.train()
            for curdir, dirs, files in os.walk(self.train_path):
                if len(files) == 0:
                    continue
                random.shuffle(files)
                for file in files:
                    cur_path = os.path.join(curdir, file)
                    train_dataset = MyDataset(path=cur_path, len_input=self.len_input)
                    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=1, drop_last=True)
                    for inputs in train_dataloader:
                        optimizer.zero_grad()
                        inputs = torch.true_divide(inputs, 255).to(rank)
                        total_loss = model(inputs, PredSteps=10, mode='train')
                        # print(total_loss)
                        total_loss.backward()
                        optimizer.step()

            # validation
            ssim, psnr, lpips = self.val_model(model)
            ssim_score.append(ssim)
            psnr_score.append(psnr)
            lpips_score.append(lpips)
            print('ssim score', ssim, 'psnr score', psnr, 'lpips', lpips,
                  'epoch: ', epoch, 'time: ', datetime.datetime.now().strftime('%H:%M:%S'))
            # print('mse_loss: ', mse_loss, 'lpips_loss: ', lpips_loss)

            if ssim + psnr / 100 - lpips > cur_loss:
                torch.save({
                    'info':info,
                    'epoch': epoch,
                    'model_state': model.state_dict(),
                    'loss': (ssim + psnr / 100 - lpips)
                }, state_path)
                cur_loss = (ssim + psnr / 100 - lpips)

            torch.save({'ssim': ssim_score, 'psnr': psnr_score, 'lpips': lpips_score}, acc_path)


if __name__ == '__main__':
    rank = 0
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(rank)
    demo = Training()
    demo.StartTraining(rank=0)


