import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MSELoss_ADPIT(object):
    def __init__(self, relative_dist=False, no_dist=False):
        super().__init__()
        self._each_loss = nn.MSELoss(reduction='none')
        self.relative_dist = relative_dist
        self.no_dist = no_dist
        self.eps = 0.001

    def _each_calc(self, output, target):
        loss = self._each_loss(output, target)

        if self.no_dist:
            loss[:, :, 3] = 0.
            loss[:, :, 7] = 0.
            loss[:, :, 11] = 0.

        elif self.relative_dist:
            loss[:, :, 3] = torch.where(target[:, :, 3] > 0., loss[:, :, 3] / (target[:, :, 3] + self.eps), loss[:, :, 3])
            loss[:, :, 7] = torch.where(target[:, :, 7] > 0., loss[:, :, 7] / (target[:, :, 7] + self.eps), loss[:, :, 7])
            loss[:, :, 11] = torch.where(target[:, :, 11] > 0., loss[:, :, 11] / (target[:, :, 11] + self.eps), loss[:, :, 11])

        return loss.mean(dim=(2))

    def __call__(self, output, target):
        target_A0 = target[:, :, 0, 0:1, :] * target[:, :, 0, 1:, :]
        target_B0 = target[:, :, 1, 0:1, :] * target[:, :, 1, 1:, :]
        target_B1 = target[:, :, 2, 0:1, :] * target[:, :, 2, 1:, :]
        target_C0 = target[:, :, 3, 0:1, :] * target[:, :, 3, 1:, :]
        target_C1 = target[:, :, 4, 0:1, :] * target[:, :, 4, 1:, :]
        target_C2 = target[:, :, 5, 0:1, :] * target[:, :, 5, 1:, :]

        target_A0A0A0 = torch.cat((target_A0, target_A0, target_A0), 2)
        target_B0B0B1 = torch.cat((target_B0, target_B0, target_B1), 2)
        target_B0B1B0 = torch.cat((target_B0, target_B1, target_B0), 2)
        target_B0B1B1 = torch.cat((target_B0, target_B1, target_B1), 2)
        target_B1B0B0 = torch.cat((target_B1, target_B0, target_B0), 2)
        target_B1B0B1 = torch.cat((target_B1, target_B0, target_B1), 2)
        target_B1B1B0 = torch.cat((target_B1, target_B1, target_B0), 2)
        target_C0C1C2 = torch.cat((target_C0, target_C1, target_C2), 2)
        target_C0C2C1 = torch.cat((target_C0, target_C2, target_C1), 2)
        target_C1C0C2 = torch.cat((target_C1, target_C0, target_C2), 2)
        target_C1C2C0 = torch.cat((target_C1, target_C2, target_C0), 2)
        target_C2C0C1 = torch.cat((target_C2, target_C0, target_C1), 2)
        target_C2C1C0 = torch.cat((target_C2, target_C1, target_C0), 2)

        output = output.reshape(output.shape[0], output.shape[1], target_A0A0A0.shape[2], target_A0A0A0.shape[3])

        pad4A = target_B0B0B1 + target_C0C1C2
        pad4B = target_A0A0A0 + target_C0C1C2
        pad4C = target_A0A0A0 + target_B0B0B1

        loss_list = [
            self._each_calc(output, target_A0A0A0 + pad4A),
            self._each_calc(output, target_B0B0B1 + pad4B),
            self._each_calc(output, target_B0B1B0 + pad4B),
            self._each_calc(output, target_B0B1B1 + pad4B),
            self._each_calc(output, target_B1B0B0 + pad4B),
            self._each_calc(output, target_B1B0B1 + pad4B),
            self._each_calc(output, target_B1B1B0 + pad4B),
            self._each_calc(output, target_C0C1C2 + pad4C),
            self._each_calc(output, target_C0C2C1 + pad4C),
            self._each_calc(output, target_C1C0C2 + pad4C),
            self._each_calc(output, target_C1C2C0 + pad4C),
            self._each_calc(output, target_C2C0C1 + pad4C),
            self._each_calc(output, target_C2C1C0 + pad4C),
        ]

        loss_stack = torch.stack(loss_list, dim=0)
        loss_min = torch.min(loss_stack, dim=0).indices

        loss = sum(
            loss_list[i] * (loss_min == i) for i in range(len(loss_list))
        ).mean()

        return loss


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class SeldModel(nn.Module):
    def __init__(self, in_feat_shape, out_shape, params):
        super().__init__()
        self.params = params
        self.nb_classes = params['unique_classes']
        self.conv_block_list = nn.ModuleList()

        for conv_cnt in range(len(params['f_pool_size'])):
            in_ch = params['nb_cnn2d_filt'] if conv_cnt else in_feat_shape[1]
            self.conv_block_list.append(ConvBlock(in_ch, params['nb_cnn2d_filt']))
            self.conv_block_list.append(nn.MaxPool2d((params['t_pool_size'][conv_cnt], params['f_pool_size'][conv_cnt])))
            self.conv_block_list.append(nn.Dropout2d(p=params['dropout_rate']))

        self.gru_input_dim = params['nb_cnn2d_filt'] * int(np.floor(in_feat_shape[-1] / np.prod(params['f_pool_size'])))
        self.gru = nn.GRU(
            input_size=self.gru_input_dim,
            hidden_size=params['rnn_size'],
            num_layers=params['nb_rnn_layers'],
            batch_first=True,
            dropout=params['dropout_rate'],
            bidirectional=True
        )

        self.mhsa_block_list = nn.ModuleList()
        self.layer_norm_list = nn.ModuleList()
        for _ in range(params['nb_self_attn_layers']):
            self.mhsa_block_list.append(
                nn.MultiheadAttention(
                    embed_dim=params['rnn_size'],
                    num_heads=params['nb_heads'],
                    dropout=params['dropout_rate'],
                    batch_first=True
                )
            )
            self.layer_norm_list.append(nn.LayerNorm(params['rnn_size']))

        self.fnn_list = nn.ModuleList()
        if params['nb_fnn_layers']:
            for i in range(params['nb_fnn_layers']):
                in_f = params['fnn_size'] if i else params['rnn_size']
                self.fnn_list.append(nn.Linear(in_f, params['fnn_size']))
        final_in = params['fnn_size'] if params['nb_fnn_layers'] else params['rnn_size']
        self.fnn_list.append(nn.Linear(final_in, out_shape[-1]))

    def forward(self, x):
        for layer in self.conv_block_list:
            x = layer(x)

        x = x.transpose(1, 2).contiguous()
        x = x.view(x.shape[0], x.shape[1], -1).contiguous()

        x, _ = self.gru(x)
        x = torch.tanh(x)
        x = x[:, :, x.shape[-1] // 2:] * x[:, :, :x.shape[-1] // 2]

        for attn, norm in zip(self.mhsa_block_list, self.layer_norm_list):
            x_attn_in = x
            x, _ = attn(x_attn_in, x_attn_in, x_attn_in)
            x = norm(x + x_attn_in)

        for layer in self.fnn_list[:-1]:
            x = layer(x)
        doa = self.fnn_list[-1](x)

        return doa
    
    