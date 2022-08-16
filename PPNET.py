import torch
import torch.nn as nn
import torch.nn.functional as F
from ConvLSTM import ConvLSTMCell

class PPNet(nn.Module):
    def __init__(self, channels, weight_factor):
        super(PPNet, self).__init__()
        self.channels = channels
        self.num_layers = len(channels)
        self.weight_factor = weight_factor
        for l in range(self.num_layers):
            # without prediction from higher level
            lstm_wohp = ConvLSTMCell(channels[l], channels[l], kernel_size=(3, 3))
            setattr(self, 'lstm_wohp{}'.format(l), lstm_wohp)
            # with prediction from higher level
            if l < self.num_layers - 1:
                lstm_whp = ConvLSTMCell(channels[l] + channels[l+1], channels[l], kernel_size=(3, 3))
                setattr(self, 'lstm_whp{}'.format(l), lstm_whp)

        for l in range(self.num_layers - 1):
            update_A = nn.Sequential(
                nn.Conv2d(3 * channels[l], channels[l+1], kernel_size=(3, 3), padding=1),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            DownPred = nn.Sequential(
                nn.Conv2d(channels[l+1], channels[l+1], kernel_size=(3, 3), padding=1),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            )
            setattr(self, 'update_A{}'.format(l), update_A)
            setattr(self, 'DownPred{}'.format(l), DownPred)


    def MakePrediction(
            self,
            cur_A,         # current input A_l^t
            cur_H,         # current hidden h_l^t and c_l^t
            higher_P,      # prediction from higher level P_{l+1}^{t+1}
            level          # current level
    ):
        if cur_A == None:
            return None, cur_H
        if higher_P == None:
            lstmcell = getattr(self, 'lstm_wohp{}'.format(level))
            cur_P, next_H = lstmcell(cur_A, cur_H)
        else:
            lstmcell = getattr(self, 'lstm_whp{}'.format(level))
            DownPred = getattr(self, 'DownPred{}'.format(level))
            higher_P = DownPred(higher_P)
            cur_P, next_H = lstmcell(torch.cat([cur_A, higher_P], dim=1), cur_H)
        return cur_P, next_H


    def UpdateInput(
            self,
            cur_P,         # current prediction P_l^t
            cur_A,         # current time-step input A_l^t
            cur_E,         # current error E_l^t
            next_A,        # next time-step input A_l^{t+1}
            level          # current level
    ):
        if cur_P == None or next_A == None:
            # no input and target for higher level and error remain unchanged
            return None, cur_E
        # make prediction error
        pos = F.relu(cur_P - next_A)
        neg = F.relu(next_A - cur_P)
        next_E = torch.cat([pos, neg], dim=1)

        update_A = getattr(self, 'update_A{}'.format(level))
        higher_A = update_A(torch.cat([cur_A, next_E], dim=1))

        return higher_A, next_E


    def ComputeLoss(self, predict, target, lambda_t, weigh_factor):
        pos = F.relu(predict - target)
        neg = F.relu(target - predict)
        error = torch.cat([pos, neg], dim=1)
        cur_loss = torch.mean(error ** 2 * weigh_factor) * lambda_t
        return cur_loss


    def forward(self, inputs, PredSteps, mode):
        '''
        param:
            inputs.size = (b, t, c, h, w)
            pred_steps: prediction steps (for long-term prediction)
            mode: 'train' or 'test'
            b: batch size
            t: time step
        '''
        assert mode in ['train', 'val', 'test'], \
            'Invalid mode, expect "train", "val" or "test", but got {}'.format(mode)

        H_seq = []                                            # for storing hidden variable of ConvLSTM
        E_seq = [None] * self.num_layers                      # for storing local errors
        list_P = []                                           # for storing local predictions
        list_A = []                                           # for storing local inputs
        predictions = []                                      # for storing predictions of the lowest level
        batch, time_step, c, high, width = inputs.size()      # extract the input size parameters
        loss = 0

        if mode == 'train':
            real_t = time_step - PredSteps  # the number of steps that using real frames as inputs
        else:
            real_t = time_step

        # initialization
        for l in range(self.num_layers):
            h_c = torch.zeros(batch, self.channels[l], high, width, device=inputs.device)               # hidden or cell
            H_seq.append((h_c, h_c))
            E_seq[l] = torch.zeros(batch, self.channels[l] * 2, high, width, device=inputs.device)
            list_P.append([None] * (real_t + PredSteps + 1))
            list_A.append([None] * (real_t + PredSteps + 1))
            width = width // 2
            high = high // 2

        for t in range(real_t):
            list_A[0][t] = inputs[:, t]

        for t in range(real_t + PredSteps-1):
            for l in reversed(range(self.num_layers)):
                cur_A = list_A[l][t]
                cur_H = H_seq[l]
                if l == self.num_layers - 1:
                    # no predictions from higher level at the highest level
                    higher_P = None
                else:
                    higher_P = list_P[l+1][t]

                cur_P, next_H = self.MakePrediction(cur_A, cur_H, higher_P, level=l)
                list_P[l][t] = cur_P
                H_seq[l] = next_H
                if l == 0 and t >= real_t - 1:
                    if mode == 'test' or mode == 'val':
                        predictions.append(F.relu(cur_P))
                    list_A[0][t+1] = F.relu(cur_P)

            if t == real_t + PredSteps - 1:
                if mode == 'test' or mode == 'val':
                    break

            for l in range(self.num_layers - 1):
                cur_P = list_P[l][t]
                cur_E = E_seq[l]
                cur_A = list_A[l][t]
                next_A = list_A[l][t+1]
                higher_A, next_E = self.UpdateInput(cur_P, cur_A, cur_E, next_A, level=l)
                E_seq[l] = next_E
                list_A[l+1][t+1] = higher_A
                # calculating loss
                if l == 0 and mode == 'train':
                    if t == 0:
                        lambda_t = 0.5
                    else:
                        lambda_t = 1.0
                    loss += self.ComputeLoss(cur_P, inputs[:, t+1], lambda_t, self.weight_factor)

        if mode == 'train':
            return loss
        else:
            return predictions



if __name__ == '__main__':
    x = torch.rand(1, 20, 3, 128, 160)
    channels = (3, 64, 64, 256, 256, 512)
    model = PPNet(channels, weight_factor=1000)
    checkpoint = model(x, PredSteps=10, mode='val')
    print(len(checkpoint) if isinstance(checkpoint, list) else checkpoint)



