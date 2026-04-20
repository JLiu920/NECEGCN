import torch
from torch import nn
import torch.nn.functional as F


class Spatial_Basic_Block(nn.Module):
    def __init__(self, in_channels, out_channels, in_channels_e, out_channels_e, max_graph_distance, A, A_e, A_T, A_a,
                 a_T, residual=False, edge_importance=True, adaptive=False, isfirst=False, **kwargs):
        super(Spatial_Basic_Block, self).__init__()
        self.isfirst = isfirst
        if not residual:
            self.residual = lambda x: 0
            self.residual2 = lambda x: 0
            self.residual3 = lambda x: 0
        elif in_channels == out_channels:
            self.residual = lambda x: x
            self.residual2 = lambda x: x
            self.residual3 = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.InstanceNorm2d(out_channels),
            )
            self.residual2 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.InstanceNorm2d(out_channels),
            )
            if self.isfirst:
                self.residual3 = nn.Sequential(
                    nn.Conv2d(2, out_channels, 1),
                    nn.InstanceNorm2d(out_channels),
                )
            else:
                self.residual3 = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    nn.InstanceNorm2d(out_channels),
                )

        self.conv = SpatialGraphConv(in_channels, out_channels, in_channels_e, out_channels_e, max_graph_distance,
                                     isfirst=self.isfirst)
        self.bn = nn.InstanceNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        if adaptive:
            self.A = nn.Parameter(A, requires_grad=True)
        else:
            self.register_buffer('A', A)
        self.edge = nn.Parameter(torch.ones_like(A), requires_grad=edge_importance)

        self.A_e = nn.Parameter(A_e, requires_grad=True)
        self.node = nn.Parameter(torch.ones_like(A_e), requires_grad=edge_importance)

        self.A_a = nn.Parameter(A_a, requires_grad=True)
        self.node2 = nn.Parameter(torch.ones_like(A_a), requires_grad=edge_importance)

        self.A_T = nn.Parameter(A_T, requires_grad=True)
        self.a_T = nn.Parameter(a_T, requires_grad=True)

    def forward(self, x, x_e, x_a):
        res_block = self.residual(x)
        res_block_e = self.residual2(x_e)
        res_block_a = self.residual3(x_a)
        x, x_e, x_a = self.conv(x, x_e, x_a, self.A * self.edge, self.A_e * self.node, self.A_a * self.node2, self.A_T,
                                self.a_T)
        x = self.bn(x)
        x_e = self.bn(x_e)
        x_a = self.bn(x_a)
        x = self.relu(x + res_block)
        x_e = self.relu(x_e + res_block_e)
        x_a = self.relu(x_a + res_block_a)
        return x, x_e, x_a


class Temporal_Basic_Block(nn.Module):
    def __init__(self, channels, temporal_window_size, stride=1, residual=False, **kwargs):
        super(Temporal_Basic_Block, self).__init__()
        padding = ((temporal_window_size - 1) // 2, 0)
        if not residual:
            self.residual = lambda x: 0
        elif stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(channels, channels, 1, (stride, 1)),
                nn.BatchNorm2d(channels),
            )
        self.conv = nn.Conv2d(channels, channels, (temporal_window_size, 1), (stride, 1), padding)
        self.bn = nn.InstanceNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, res_module):
        res_block = self.residual(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x + res_block + res_module)
        return x


class Temporal_MultiScale_Block(nn.Module):
    def __init__(self, out_channels, kernel_size=3, stride=1, residual=True, dilations=[1, 2], residual_kernel_size=1,
                 **kwargs):

        super().__init__()
        in_channels = out_channels
        self.num_branches = len(dilations) + 2
        branch_channels = out_channels // self.num_branches
        if type(kernel_size) == list:
            assert len(kernel_size) == len(dilations)
        else:
            kernel_size = [kernel_size] * len(dilations)
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    branch_channels,
                    kernel_size=1,
                    padding=0),
                nn.InstanceNorm2d(branch_channels),
                nn.ReLU(inplace=True),
                TemporalConv(
                    branch_channels,
                    branch_channels,
                    kernel_size=ks,
                    stride=stride,
                    dilation=dilation),
            )
            for ks, dilation in zip(kernel_size, dilations)
        ])
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
            nn.InstanceNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0)),
            nn.InstanceNorm2d(branch_channels)
        ))
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0, stride=(stride, 1)),
            nn.InstanceNorm2d(branch_channels)
        ))
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(in_channels, out_channels, kernel_size=residual_kernel_size, stride=stride)

    def forward(self, x, res_module):
        res = self.residual(x)
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)
        out = torch.cat(branch_outs, dim=1)
        out += res + res_module
        return out


class TimeAwareGraphConvolution(nn.Module):

    def __init__(self, in_feat_v, out_feat_v, in_feat_e, out_feat_e, layer=1):
        super().__init__()
        self.node_layer = layer

        if self.node_layer == 1:
            self.W_v = nn.Parameter(torch.Tensor(in_feat_v, out_feat_v))
            self.P_e = nn.Parameter(torch.Tensor(in_feat_v, 1))
        elif self.node_layer == 2:
            self.W_e = nn.Parameter(torch.Tensor(in_feat_v, in_feat_v))
            self.P_v = nn.Parameter(torch.Tensor(in_feat_v, 1))
        elif self.node_layer == 3:
            self.W_v = nn.Parameter(torch.Tensor(in_feat_v, out_feat_v))
            self.P_a = nn.Parameter(torch.Tensor(in_feat_v, 1))
        elif self.node_layer == 4:
            self.W_a = nn.Parameter(torch.Tensor(in_feat_v, in_feat_v))
            self.P_v = nn.Parameter(torch.Tensor(in_feat_v, 1))

        self.reset_parameters()

    def reset_parameters(self):
        if self.node_layer == 1:
            nn.init.xavier_uniform_(self.W_v)
            nn.init.xavier_uniform_(self.P_e)
        elif self.node_layer == 2:
            nn.init.xavier_uniform_(self.W_e)
            nn.init.xavier_uniform_(self.P_v)
        elif self.node_layer == 3:
            nn.init.xavier_uniform_(self.W_v)
            nn.init.xavier_uniform_(self.P_a)
        elif self.node_layer == 4:
            nn.init.xavier_uniform_(self.W_a)
            nn.init.xavier_uniform_(self.P_v)

    def forward(self, H_v, H_e, H_a, adj_v, adj_e, adj_a, T, T_a):
        H_v = H_v.permute(0, 2, 3, 1)
        H_e = H_e.permute(0, 2, 3, 1)
        H_a = H_a.permute(0, 2, 3, 1)
        if self.node_layer == 1:
            res_H = H_v
            Hv_trans = torch.einsum('btnd,dh->btnh', H_v, self.W_v)
            att_e = torch.einsum('bted,de->bte', H_e, self.P_e)
            att_e = torch.sigmoid(att_e).unsqueeze(-1)
            T_mask = T.to_dense().unsqueeze(0).unsqueeze(0)
            prop = torch.einsum('bave,btea->btve', T_mask, att_e)
            prop = torch.einsum('btnk,btke->btne', prop, T_mask.transpose(-1, -2))
            adj = adj_v.to_dense() * prop
            output = torch.einsum('btnm,btmh->btnh', adj, Hv_trans)
        elif self.node_layer == 2:
            res_H = H_e
            He_trans = torch.einsum('bted,dh->bteh', H_e, self.W_e)
            att_v = torch.einsum('btnd,dv->btn', H_v, self.P_v)
            att_v = torch.sigmoid(att_v).unsqueeze(-1)
            T_t = T.to_dense().t().unsqueeze(0).unsqueeze(0)
            prop = torch.einsum('baev,btva->btev', T_t, att_v)
            prop = torch.einsum('btek,btkn->bten', prop, T_t.transpose(-1, -2))
            adj = adj_e.to_dense()
            output = torch.einsum('btem,btmh->bteh', adj * prop, He_trans)
        elif self.node_layer == 3:
            res_H = H_v
            Hv_trans = torch.einsum('btnd,dh->btnh', H_v, self.W_v)
            att_a = torch.einsum('bted,de->bte', H_a, self.P_a)
            att_a = torch.sigmoid(att_a).unsqueeze(-1)
            T_mask = T_a.to_dense().unsqueeze(0).unsqueeze(0)
            prop = torch.einsum('bave,btea->btve', T_mask, att_a)
            prop = torch.einsum('btnk,btke->btne', prop, T_mask.transpose(-1, -2))
            adj = adj_v.to_dense() * prop
            output = torch.einsum('btnm,btmh->btnh', adj, Hv_trans)
        elif self.node_layer == 4:
            res_H = H_a
            Ha_trans = torch.einsum('bted,dh->bteh', H_a, self.W_a)
            att_v = torch.einsum('btnd,dv->btn', H_v, self.P_v)
            att_v = torch.sigmoid(att_v).unsqueeze(-1)
            T_t = T_a.to_dense().t().unsqueeze(0).unsqueeze(0)
            prop = torch.einsum('baev,btva->btev', T_t, att_v)
            prop = torch.einsum('btek,btkn->bten', prop, T_t.transpose(-1, -2))
            adj = adj_a.to_dense()
            output = torch.einsum('btem,btmh->bteh', adj * prop, Ha_trans)
        out1 = (output + res_H).permute(0, 3, 1, 2)
        return out1


class StandConvolution(nn.Module):
    def __init__(self, dims, dropout):
        super(StandConvolution, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.conv = nn.Sequential(
            nn.Conv2d(dims[0], dims[3], kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(dims[3]),

        )

    def forward(self, x):
        x_tmp = self.conv(x)
        x_tmp = x_tmp
        return x_tmp + x

class SpatialGraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, in_channels_e, out_channels_e, max_graph_distance, isfirst=False):
        super(SpatialGraphConv, self).__init__()
        self.isfirst = isfirst
        self.s_kernel_size = max_graph_distance + 1
        self.gcn = nn.Conv2d(in_channels, out_channels * self.s_kernel_size, 1)
        self.gcn_e = nn.Conv2d(in_channels, out_channels * self.s_kernel_size, 1)
        if self.isfirst:
            self.gcn_a = nn.Conv2d(2, out_channels * self.s_kernel_size, 1)
        else:
            self.gcn_a = nn.Conv2d(in_channels, out_channels * self.s_kernel_size, 1)

        self.gcn2 = nn.Conv2d(out_channels * self.s_kernel_size, out_channels * self.s_kernel_size, kernel_size=3,
                              stride=1, padding=1)
        self.gcn2_e = nn.Conv2d(out_channels * self.s_kernel_size, out_channels * self.s_kernel_size, kernel_size=3,
                                stride=1, padding=1)
        self.gcn2_a = nn.Conv2d(out_channels * self.s_kernel_size, out_channels * self.s_kernel_size, kernel_size=3,
                                stride=1, padding=1)

        self.gc1_node = TimeAwareGraphConvolution(out_channels * self.s_kernel_size,
                                                  out_channels * self.s_kernel_size,
                                                  out_channels * self.s_kernel_size,
                                                  out_channels * self.s_kernel_size, layer=1)
        self.gc2_node = TimeAwareGraphConvolution(out_channels * self.s_kernel_size,
                                                  out_channels * self.s_kernel_size,
                                                  out_channels * self.s_kernel_size,
                                                  out_channels * self.s_kernel_size, layer=3)
        self.gc1_edge = TimeAwareGraphConvolution(out_channels * self.s_kernel_size,
                                                  out_channels * self.s_kernel_size,
                                                  out_channels * self.s_kernel_size,
                                                  out_channels * self.s_kernel_size, layer=2)
        self.gc1_angle = TimeAwareGraphConvolution(out_channels * self.s_kernel_size,
                                                   out_channels * self.s_kernel_size,
                                                   out_channels * self.s_kernel_size,
                                                   out_channels * self.s_kernel_size, layer=4)
        self.gc3_node = TimeAwareGraphConvolution(out_channels * self.s_kernel_size,
                                                  out_channels * self.s_kernel_size,
                                                  out_channels * self.s_kernel_size,
                                                  out_channels * self.s_kernel_size, layer=1)
        self.gc4_node = TimeAwareGraphConvolution(out_channels * self.s_kernel_size,
                                                  out_channels * self.s_kernel_size,
                                                  out_channels * self.s_kernel_size,
                                                  out_channels * self.s_kernel_size, layer=3)
        self.gc2_edge = TimeAwareGraphConvolution(out_channels * self.s_kernel_size,
                                                  out_channels * self.s_kernel_size,
                                                  out_channels * self.s_kernel_size,
                                                  out_channels * self.s_kernel_size, layer=2)
        self.gc2_angle = TimeAwareGraphConvolution(out_channels * self.s_kernel_size,
                                                   out_channels * self.s_kernel_size,
                                                   out_channels * self.s_kernel_size,
                                                   out_channels * self.s_kernel_size, layer=4)

    def forward(self, x, x_e, x_a, A, A_e, A_T, A_a, a_T):

        x = self.gcn(x)
        x_e = self.gcn_e(x_e)
        x_a = self.gcn_a(x_a)

        x_e = self.gc1_edge(x, x_e, x_a, A, A_e, A_a, A_T, a_T)
        x_e = F.relu(x_e)
        x = self.gc1_node(x, x_e, x_a, A, A_e, A_a, A_T, a_T)
        x = F.relu(x)
        x_a = self.gc1_angle(x, x_e, x_a, A, A_e, A_a, A_T, a_T)
        x_a = F.relu(x_a)
        x = self.gc2_node(x, x_e, x_a, A, A_e, A_a, A_T, a_T)
        x = F.relu(x)

        x_e = self.gc2_edge(x, x_e, x_a, A, A_e, A_a, A_T, a_T)
        x_e = F.relu(x_e)
        x = self.gc3_node(x, x_e, x_a, A, A_e, A_a, A_T, a_T)
        x = F.relu(x)
        x_a = self.gc2_angle(x, x_e, x_a, A, A_e, A_a, A_T, a_T)
        x_a = F.relu(x_a)
        x = self.gc4_node(x, x_e, x_a, A, A_e, A_a, A_T, a_T)
        x = F.relu(x)

        x = self.gcn2(x)
        x_e = self.gcn2_e(x_e)
        x_a = self.gcn2_a(x_a)
        return x, x_e, x_a


class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(TemporalConv, self).__init__()
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1))

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x
