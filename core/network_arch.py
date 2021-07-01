# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import torch
from torch import nn
import numpy as np
import pointnet2_ops.pointnet2_modules as pointnet2
from core.utils import *
import torch.nn.functional as F
from torch.distributions import Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -10
epsilon = 1e-6


def create_encoder(
    model_scale, pointnet_radius, pointnet_nclusters, input_dim=0,
    feature_option=0, traj_net=False, group_norm=True ):
    global_pc_dim_list = [64, 256, 1024]
    local_pc_dim_list  = [32, 128, 512]
    output_dim = int(global_pc_dim_list[-1] + model_scale * 512)

    if  feature_option == 0:
        model = base_network(pointnet_radius, pointnet_nclusters, model_scale, input_dim)
    elif feature_option == 2:
        model =  base_network_new_large(model_scale, input_dim, group_norm=group_norm)
    elif feature_option == 4:
        model =  base_point_net_large(model_scale, input_dim, group_norm=group_norm) # npoint_2=128

    print('=== net option: {} trainable param: {}  === '.format(feature_option, [count_parameters(m) for m in model]))
    if traj_net:
        model = nn.ModuleList([RobotMLP(input_dim, global_pc_dim_list, gn=group_norm), model])
        return model, output_dim
    return model

def pointnet_encode(encoder, xyz, xyz_features, vis=False):
    l_xyz = [xyz]
    if type(encoder) is not nn.ModuleList:
        return encoder(xyz_features)

    for module in encoder[0]:
        xyz, xyz_features = module(xyz, xyz_features)
        l_xyz.append(xyz)

    return encoder[1](xyz_features.squeeze(-1))

def base_network(pointnet_radius, pointnet_nclusters, scale, in_features, group_norm=False):
    sa1_module = pointnet2.PointnetSAModule(
        npoint=pointnet_nclusters,
        radius=pointnet_radius,
        nsample=64,
        mlp=[in_features, int(64 * scale), int(64 * scale), int(128 * scale)],
    )
    sa2_module = pointnet2.PointnetSAModule(
        npoint=32,
        radius=0.04,
        nsample=128,
        mlp=[int(128 * scale), int(128 * scale), int(128 * scale), int(256 * scale)],
    )

    sa3_module = pointnet2.PointnetSAModule(
        mlp=[int(256 * scale), int(256 * scale), int(256 * scale), int(512 * scale)]
    )

    sa_modules = nn.ModuleList([sa1_module, sa2_module, sa3_module])
    fc_layer = nn.Sequential(
        nn.Linear(int(512 * scale), int(1024 * scale)),
        nn.BatchNorm1d(int(1024 * scale)),
        nn.ReLU(True),
        nn.Linear(int(1024 * scale), int(512 * scale)),
        nn.BatchNorm1d(int(512 * scale)),
        nn.ReLU(True),
    )
    return nn.ModuleList([sa_modules, fc_layer])

def base_network_new_large(scale, in_features,
                           npoints=[512, 256, 128], sample_npoint_list=[128,128,64], radius=[0.1, 0.2, 0.4], layer1=False,
                           group_norm=True ):
    sample_npoint_list = [64] * len(sample_npoint_list)

    sa0_module = pointnet2.PointnetSAModule(
        npoint=npoints[0],
        radius=radius[0],
        nsample=sample_npoint_list[0],
        mlp=[in_features, 32, 32, 64],
        bn=not group_norm
    )

    sa1_module = pointnet2.PointnetSAModule(
        npoint=npoints[1],
        radius=radius[1],
        nsample=sample_npoint_list[1],
        mlp=[64, 64, 64, 128],
        bn=not group_norm
    )

    sa2_module = pointnet2.PointnetSAModule(
        npoint=npoints[2], # 64
        radius=radius[2],    # 0.2
        nsample=sample_npoint_list[2],     # 64
        mlp=[128, 128, 128, 256],
        bn=not group_norm
    )

    sa3_module = pointnet2.PointnetSAModule(
        mlp=[256, 256, 256, 512],
        bn=not group_norm
    )

    sa_modules = nn.ModuleList([sa0_module, sa1_module, sa2_module, sa3_module ])
    fc_layer = nn.Sequential(
            nn.Linear(512, int(512 * scale)),
            nn.GroupNorm(16, int(512 * scale)),
            nn.ReLU(True),
            )
    return nn.ModuleList([sa_modules, fc_layer])


def base_point_net_large(  scale, in_features,
                           npoints=[512, 128], sample_npoint_list=[128, 128], radius=[0.05, 0.3],
                           group_norm=True, layer_num=2 ):

    layer_2_input_dim = in_features
    sample_npoint_list = [128] * len(sample_npoint_list)

    sa1_module = pointnet2.PointnetSAModule(
        npoint=npoints[0],
        radius=radius[0],
        nsample=sample_npoint_list[0],
        mlp=[in_features, 64, 64, 64],
        bn=not group_norm
    )

    sa2_module = pointnet2.PointnetSAModule(
        npoint=npoints[1],
        radius=radius[1],
        nsample=sample_npoint_list[1],
        mlp=[layer_2_input_dim, 128, 128, 256],
        bn=not group_norm
    )
    sa3_module = pointnet2.PointnetSAModule(
        mlp=[256, 512, 512, 1024],
        bn=not group_norm
    )

    sa_modules = nn.ModuleList([sa2_module, sa3_module])
    fc_layer = nn.Sequential(
    nn.Linear(1024, int(1024 * scale)),
    nn.GroupNorm(16, int(1024 * scale)),
    nn.ReLU(True),
    nn.Linear(int(1024 * scale), int(512 * scale)),
    nn.GroupNorm(16, int(512 * scale)),
    nn.ReLU(True),
    )
    return nn.ModuleList([sa_modules, fc_layer])


class Identity(nn.Module):
    def forward(self, input):
        return input


def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class GaussianPolicy(nn.Module):
    def __init__(
        self,
        num_inputs,
        num_actions,
        hidden_dim,
        action_space=None,
        extra_pred_dim=0,
        uncertainty=False,
        config=None
    ):
        super(GaussianPolicy, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.uncertainty = uncertainty

        self.extra_pred_dim = extra_pred_dim
        self.mean = nn.Linear(hidden_dim, num_actions)
        self.extra_pred = nn.Linear(hidden_dim, self.extra_pred_dim)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)
        self.apply(weights_init_)
        self.action_space = action_space

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.0).cuda()
            self.action_bias = torch.tensor(0.0).cuda()
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.0
            ).cuda()
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.0
            ).cuda()

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean(x)
        extra_pred = self.extra_pred(x)

        if self.extra_pred_dim >= 7:
            extra_pred = torch.cat(
                (F.normalize(extra_pred[:, :4], p=2, dim=-1), extra_pred[:, 4:]), dim=-1)

        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)

        return mean, log_std, extra_pred

    def sample(self, state):
        mean, log_std, extra_pred = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        if self.action_space is not None:
            y_t = torch.tanh(x_t)
            action = y_t * self.action_scale + self.action_bias
        else:
            y_t = x_t
            action = x_t

        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        if self.action_space is not None:
            mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return mean, log_prob, action, extra_pred


    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)


class QNetwork(nn.Module):
    def __init__(
        self,
        num_inputs,
        num_actions,
        hidden_dim,
        extra_pred_dim=0,
    ):
        super(QNetwork, self).__init__()

        self.num_actions = num_actions
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)
        self.extra_pred_dim = extra_pred_dim

        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)
        if self.extra_pred_dim > 0:
            self.linear7 = nn.Linear(num_inputs , hidden_dim)
            self.linear8 = nn.Linear(hidden_dim, hidden_dim)
            self.extra_pred = nn.Linear(hidden_dim, self.extra_pred_dim)
        self.apply(weights_init_)

    def forward(self, state, action=None):
        x3 = None
        if action is not None:
            xu = torch.cat([state, action], 1)
        else:
            xu = state
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        if self.extra_pred_dim > 0:
            if self.num_actions > 0: # state only
                state = state[:,:-self.num_actions]
            x3 = F.relu(self.linear7(state))
            x3 = F.relu(self.linear8(x3))
            x3 = self.extra_pred(x3)
            if self.extra_pred_dim >= 7: # normalize quaternion
                x3 = torch.cat((F.normalize(x3[:, :4], p=2, dim=-1), x3[:, 4:]), dim=-1)
        return x1, x2, x3
