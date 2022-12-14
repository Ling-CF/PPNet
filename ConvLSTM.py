import torch.nn as nn
import torch

class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, output_dim, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = output_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2
        self.bias = bias
        self.conv1 = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                               out_channels=4 * self.hidden_dim,
                               kernel_size=self.kernel_size,
                               padding=self.padding,
                               bias=self.bias)
        self.conv2 = nn.Conv2d(in_channels=4 * self.hidden_dim,
                               out_channels=4 * self.hidden_dim,
                               kernel_size=self.kernel_size,
                               padding=self.padding,
                               bias=self.bias)
        self.out_conv = nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=kernel_size, padding=self.padding)
        self.relu = nn.LeakyReLU(inplace=True)


    def forward(self, input_tensor, cur_state):

        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined = self.conv1(combined)
        combined = self.relu(combined)
        combined = self.conv2(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined, self.hidden_dim, dim=1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return self.relu(self.out_conv(h_next)), (h_next, c_next)

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))