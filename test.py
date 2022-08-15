import os
import numpy as np
import torch
from PPNET import PPNet
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from utils import MyDataset, util_of_lpips, ShowImages


def test_model(
        rank,
        val_data    =   './datasets/KTH/val',        						# replace it with the path where you store your testing dataset
        state_path  =   './models/KTH/model_KTH.pt',                        # replace it with the path where you store your model
        num_pre     =   30,                                                 # the number of time steps in the long term prediction
        show_img    =   True,                                               # bool: show image or not
):
    model = PPNet(channels=(3, 64, 64, 256, 256, 512), weight_factor=1000)
    model = model.to(rank)
    checkpoint = torch.load(state_path, map_location='cuda:{}'.format(rank))
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    all_ssim = []
    all_psnr = []
    all_lpips = []
    lpips_loss = util_of_lpips('alex')

    with torch.no_grad():
        for curdir, dirs, files in os.walk(val_data):
            if len(files) == 0:
                continue
            files.sort()
            # print(files)
            for file in files:
                cur_path = os.path.join(curdir, file)
                test_dataset = MyDataset(path=cur_path, len_input=10, interval=1)
                test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=1, drop_last=True)
                test_dataloader = list(test_dataloader)
                for item in range(0, len(test_dataloader), 4):
                    if item > len(test_dataloader) - 4:
                        break
                    inputs = torch.true_divide(test_dataloader[item], 255).to(rank)
                    targets = torch.true_divide(torch.cat(test_dataloader[item+1:item+4], dim=1), 255).to(rank)
                    ssim_score = []
                    psnr_score = []
                    lpips_score = []
                    predictions = model(inputs, PredSteps=num_pre, mode='test')

                    if show_img:
                        ShowImages(predictions, targets)

                    for t in range(num_pre):
                        target = targets[:, t]
                        predict = predictions[t]
                        lpips = lpips_loss.calc_lpips(predict, target)
                        lpips_score.append(lpips.item())
                        target = target.cpu().squeeze().numpy()
                        target = np.transpose(target, (1, 2, 0))
                        predict = predict.data.cpu().numpy().squeeze()
                        predict = np.transpose(predict, (1,2,0))
                        (ssim, diff) = structural_similarity(target, predict, win_size=None, multichannel=True, data_range=1.0,
                                                              full=True)
                        psnr = peak_signal_noise_ratio(target, predict, data_range=1.0)
                        psnr_score.append(psnr)
                        ssim_score.append(ssim)
                    # print(ssim_score, '\n', psnr_score, '\n', lpips_score)
                    all_ssim.append(ssim_score)
                    all_psnr.append(psnr_score)
                    all_lpips.append(lpips_score)

    all_ssim = np.array(all_ssim)
    mean_ssim = np.mean(all_ssim, axis=0)
    all_psnr = np.array(all_psnr)
    mean_psnr = np.mean(all_psnr, axis=0)
    all_lpips = np.array(all_lpips)
    mean_lpips = np.mean(all_lpips, axis=0) * 100
    print('mean ssim: ', '\n', mean_ssim, '\n', '0-10: ', np.mean(mean_ssim[:10]), '\n', '0-30: ', np.mean(mean_ssim))
    print("mean psnr: ", '\n', mean_psnr, '\n', '0-10: ', np.mean(mean_psnr[:10]), '\n', '0-30: ', np.mean(mean_psnr))
    print("mean lpips: ", '\n', mean_lpips, '\n', '0-10: ', np.mean(mean_lpips[:10]), '\n', '0-30: ',np.mean(mean_lpips))

    ###################################################################
    # saving the test results
    # cla = str(input('classification name: '))
    # ssim_name = './predictions/KTH/metrics/{}_ssim.npy'.format(cla)
    # psnr_name = './predictions/KTH/metrics/{}_psnr.npy'.format(cla)
    # lpips_name = './predictions/KTH/metrics/{}_lpips.npy'.format(cla)
    # np.save(ssim_name, mean_ssim)
    # np.save(psnr_name, mean_psnr)
    # np.save(lpips_name, mean_lpips)





if __name__ == '__main__':
    rank = 0
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(rank)
    print('start testing')
    test_model(rank=0)
