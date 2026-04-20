import logging, torch
from torch import nn
import numpy as np


class ResGCN_Module(nn.Module):
    def __init__(self, in_channels, out_channels, in_channels_e, out_channels_e, spatial_block, temporal_block, A, A_e,
                 A_T, A_a, a_T, initial=False, stride=1, kernel_size=[9, 2], isfirst=False, **kwargs):
        super(ResGCN_Module, self).__init__()
        self.isfirst = isfirst
        if not len(kernel_size) == 2:
            logging.info('')
            logging.error('Error: Please check whether len(kernel_size) == 2')
            raise ValueError()
        if not kernel_size[0] % 2 == 1:
            logging.info('')
            logging.error('Error: Please check whether kernel_size[0] % 2 == 1')
            raise ValueError()
        temporal_window_size, max_graph_distance = kernel_size

        if initial:
            module_res, block_res = False, False
        elif spatial_block == 'Basic' and temporal_block == 'Basic':
            module_res, block_res = True, False
        else:
            module_res, block_res = False, True

        if not module_res:
            self.residual = lambda x: 0
            self.residual2 = lambda x: 0
            self.residual3 = lambda x: 0
        elif stride == 1 and in_channels == out_channels:
            self.residual = lambda x: x
            self.residual2 = lambda x: x
            self.residual3 = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, (stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
            self.residual2 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, (stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
            self.residual3 = nn.Sequential(
                nn.Conv2d(2, out_channels, 1, (stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        spatial_block = U.import_class('src.model.NECEGCN.blocks.Spatial_{}_Block'.format(spatial_block))
        temporal_block = U.import_class('src.model.NECEGCN.blocks.Temporal_{}_Block'.format(temporal_block))
        if initial and 'adaptive' in kwargs:
            kwargs['adaptive'] == False
        self.scn = spatial_block(in_channels, out_channels, in_channels_e, out_channels_e, max_graph_distance, A, A_e,
                                 A_T, A_a, a_T, block_res, isfirst=self.isfirst, **kwargs)
        self.tcn = temporal_block(out_channels, temporal_window_size, stride, block_res, **kwargs)
        self.tcn_e = temporal_block(out_channels, temporal_window_size, stride, block_res, **kwargs)
        self.tcn_a = temporal_block(out_channels, temporal_window_size, stride, block_res, **kwargs)

    def forward(self, x):
        B, C, T, V = x.size()
        if self.isfirst:
            x_v = x[:, :(C - 2) // 2, :, :]
            x_e = x[:, (C - 2) // 2:-2, :, :]
            x_a = x[:, -2:C, :, :]
        else:
            x_v = x[:, :C // 3, :, :]
            x_e = x[:, C // 3: C // 3 * 2, :, :]
            x_a = x[:, C // 3 * 2:C, :, :]
        x_v, x_e, x_a = self.scn(x_v, x_e, x_a)
        x_v = self.tcn(x_v, self.residual(x_v))
        x_e = self.tcn_e(x_e, self.residual2(x_e))
        x_a = self.tcn_a(x_a, self.residual3(x_a))
        x = torch.cat((x_v, x_e, x_a), dim=1)
        return x, x_v, x_e


class AttGCN_Module(nn.Module):
    def __init__(self, in_channels, out_channels, in_channels_e, out_channels_e, spatial_block, temporal_block, A, A_e,
                 A_T, A_a, a_T, attention, stride=1, kernel_size=[9, 2], **kwargs):
        super(AttGCN_Module, self).__init__()

        if not len(kernel_size) == 2:
            logging.info('')
            logging.error('Error: Please check whether len(kernel_size) == 2')
            raise ValueError()
        if not kernel_size[0] % 2 == 1:
            logging.info('')
            logging.error('Error: Please check whether kernel_size[0] % 2 == 1')
            raise ValueError()
        temporal_window_size, max_graph_distance = kernel_size

        if spatial_block == 'Basic' and temporal_block == 'Basic':
            module_res, block_res = True, False
        else:
            module_res, block_res = False, True

        if not module_res:
            self.residual = lambda x: 0
            self.residual2 = lambda x: 0
            self.residual3 = lambda x: 0
        elif stride == 1 and in_channels == out_channels:
            self.residual = lambda x: x
            self.residual2 = lambda x: x
            self.residual3 = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, (stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
            self.residual2 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, (stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
            self.residual3 = nn.Sequential(
                nn.Conv2d(2, out_channels, 1, (stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        spatial_block = U.import_class('src.model.NECEGCN.blocks.Spatial_{}_Block'.format(spatial_block))
        temporal_block = U.import_class('src.model.NECEGCN.blocks.Temporal_{}_Block'.format(temporal_block))
        self.scn = spatial_block(in_channels // 2, out_channels // 2, in_channels_e // 2, out_channels_e // 2,
                                 max_graph_distance, A, A_e,
                                 A_T, A_a, a_T, block_res, **kwargs)
        self.tcn = temporal_block(out_channels // 2, temporal_window_size, stride, block_res, **kwargs)
        self.tcn_e = temporal_block(out_channels // 2, temporal_window_size, stride, block_res, **kwargs)
        self.tcn_a = temporal_block(out_channels // 2, temporal_window_size, stride, block_res, **kwargs)

        self.att = CrossGuidedAttention(channels=out_channels // 2, T=32, N=50)

    def forward(self, x):
        B, C, T, V = x.size()
        x_v = x[:, :C // 3, :, :]
        x_e = x[:, C // 3: C // 3 * 2, :, :]
        x_a = x[:, C // 3 * 2:C, :, :]
        x_v, x_e, x_a = self.scn(x_v, x_e, x_a)
        x_v = self.tcn(x_v, self.residual(x_v))
        x_e = self.tcn_e(x_e, self.residual2(x_e))
        x_a = self.tcn_a(x_a, self.residual3(x_a))
        x_v, x_e, x_a = self.att(x_v, x_e, x_a)
        x = torch.cat((x_v, x_e, x_a), dim=1)
        return x, x_v, x_e, x_a


class CrossGuidedAttention(nn.Module):
    def __init__(self, channels, T, N, reduction=8):
        super().__init__()
        self.channels = channels
        self.T = T
        self.N = N

        self.temporal_fc1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Softmax(dim=2)
        )
        self.spatial_fc1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Softmax(dim=3)
        )
        self.temporal_fc2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Softmax(dim=2)
        )
        self.spatial_fc2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Softmax(dim=3)
        )
        self.temporal_fc3 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Softmax(dim=2)
        )
        self.spatial_fc3 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.ReLU(),
            nn.Softmax(dim=3)
        )
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()

    def forward(self, x1, x2, x3):
        B, C, T, N = x1.size()

        def guide_attention(source, target1, target2, index):

            if index == 1:
                s1 = source
                avg_t1 = nn.AdaptiveAvgPool2d((T, 1))(target1)
                avg_t2 = nn.AdaptiveAvgPool2d((T, 1))(target2)
                avg_t = avg_t1 + avg_t2
                attn_t = self.temporal_fc1(avg_t)

                avg_s1 = nn.AdaptiveAvgPool2d((1, N))(target1)
                avg_s2 = nn.AdaptiveAvgPool2d((1, N))(target2)
                avg_s = avg_s1 + avg_s2
                attn_s = self.spatial_fc1(avg_s)
                x_s = torch.einsum('bchw,bcha,bcaw->bchw', s1, attn_t, attn_s)

            elif index == 2:
                s1 = source
                avg_t1 = nn.AdaptiveAvgPool2d((T, 1))(target1)
                avg_t2 = nn.AdaptiveAvgPool2d((T, 1))(target2)
                avg_t = avg_t1 + avg_t2
                attn_t = self.temporal_fc2(avg_t)
                avg_s1 = nn.AdaptiveAvgPool2d((1, N))(target1)
                avg_s2 = nn.AdaptiveAvgPool2d((1, N))(target2)
                avg_s = avg_s1 + avg_s2
                attn_s = self.spatial_fc2(avg_s)
                x_s = torch.einsum('bchw,bcha,bcaw->bchw', s1, attn_t, attn_s)
            else:

                s1 = source
                avg_t1 = nn.AdaptiveAvgPool2d((T, 1))(target1)
                avg_t2 = nn.AdaptiveAvgPool2d((T, 1))(target2)
                avg_t = avg_t1 + avg_t2
                attn_t = self.temporal_fc3(avg_t)
                avg_s1 = nn.AdaptiveAvgPool2d((1, N))(target1)
                avg_s2 = nn.AdaptiveAvgPool2d((1, N))(target2)
                avg_s = avg_s1 + avg_s2
                attn_s = self.spatial_fc3(avg_s)
                x_s = torch.einsum('bchw,bcha,bcaw->bchw', s1, attn_t, attn_s)
            return x_s
        out1 = guide_attention(x1, x2, x3, 1)
        out2 = guide_attention(x2, x1, x3, 2)
        out3 = guide_attention(x3, x1, x2, 3)
        return x1 + self.relu(self.bn(x1 * out1)), x2 + self.relu(self.bn(x2 * out2)), x3 + self.relu(
            self.bn(x3 * out3))
