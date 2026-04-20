import torch
from torch import nn

from .modules import ResGCN_Module


class ResGCN_Input_Branch(nn.Module):
    def __init__(self, structure, spatial_block, temporal_block, num_channel, A, A_e, A_T, A_a, a_T, **kwargs):
        super(ResGCN_Input_Branch, self).__init__()

        module_list = [
            ResGCN_Module(num_channel, 64, num_channel, 64, 'Basic', temporal_block, A, A_e, A_T, A_a, a_T,
                          isfirst=True,
                          initial=True, **kwargs)]
        module_list += [ResGCN_Module(64, 64, 64, 64, 'Basic', temporal_block, A, A_e, A_T, A_a, a_T, initial=True, **kwargs)
            for _ in range(structure[0] - 1)]
        module_list += [ResGCN_Module(64, 64, 64, 64, 'Basic', temporal_block, A, A_e, A_T, A_a, a_T, **kwargs) for _ in
                        range(structure[1] - 1)]
        module_list += [ResGCN_Module(64, 64, 64, 64, 'Basic', temporal_block, A, A_e, A_T, A_a, a_T, **kwargs)]

        self.bn = nn.InstanceNorm2d(num_channel * 2)
        self.layers = nn.ModuleList(module_list)

    def forward(self, x):
        N, C, T, V, M = x.size()
        x = self.bn(x.permute(0, 4, 1, 2, 3).contiguous().view(N * M, C, T, V))
        for layer in self.layers:
            x, x_v, x_e = layer(x)

        return x, x_v, x_e


class NECEGCN(nn.Module):
    def __init__(self, module, structure, spatial_block, temporal_block, data_shape, num_class, A, A_e, A_T, A_a, a_T,
                 **kwargs):
        super(NECEGCN, self).__init__()

        num_input, num_channel, _, _, _ = data_shape
        num_input = num_input - 1
        self.input_branches = ResGCN_Input_Branch(structure, spatial_block, temporal_block,
                                                  num_channel * num_input // 2, A, A_e, A_T, A_a, a_T,
                                                  **kwargs)
        module_list = [
            module(32 * num_input, 64, 32 * num_input, 64, spatial_block, temporal_block, A, A_e, A_T, A_a, a_T,
                   stride=2,
                   **kwargs)]
        module_list += [module(64, 64, 64, 64, spatial_block, temporal_block, A, A_e, A_T, A_a, a_T, **kwargs) for _ in
                        range(structure[2] - 1)]
        module_list += [
            module(64, 64, 64, 64, spatial_block, temporal_block, A, A_e, A_T, A_a, a_T, stride=2, **kwargs)]
        module_list += [module(64, 64, 64, 64, spatial_block, temporal_block, A, A_e, A_T, A_a, a_T, **kwargs) for _ in
                        range(structure[3] - 1)]
        self.main_stream = nn.ModuleList(module_list)

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.fcnv = nn.Linear(32, num_class)
        self.fcne = nn.Linear(32, num_class)
        self.fcna = nn.Linear(32, num_class)

    def forward(self, x):
        N, I, C, T, V, M = x.size()
        x = torch.cat(
            (x[:, 0, :, :, :, :], x[:, 1, :, :, :, :], x[:, 2, :, :, :, :], x[:, 3, :, :, :, :], x[:, 4, :2, :, :, :]),
            dim=1)
        x, x_v, x_e = self.input_branches(x)
        for layer in self.main_stream:
            x, x_v, x_e, x_a = layer(x)

        _, C, T, V = x.size()

        x_v = self.global_pooling(x[:, :C // 3, :, :])
        x_v = x_v.view(N, M, -1).mean(dim=1)
        fx_v = x_v
        x_v = self.fcnv(x_v)
        x_e = self.global_pooling(x[:, C // 3:C // 3 * 2, :, :])
        x_e = x_e.view(N, M, -1).mean(dim=1)
        fx_e = x_e
        x_e = self.fcne(x_e)
        x_a = self.global_pooling(x[:, C // 3 * 2:, :, :])
        x_a = x_a.view(N, M, -1).mean(dim=1)
        fx_a = x_a
        x_a = self.fcna(x_a)
        feature = torch.cat((fx_v, fx_e, fx_a), dim=-1)
        return x_v + x_e + x_a, feature





