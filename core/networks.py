# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
import torch
from torch import nn
import numpy as np
from core.utils import *
import torch.nn.functional as F
from torch.distributions import Normal
from core.network_arch import *


class PointTrajLatentNet(nn.Module):
    def __init__(
        self,
        input_dim=3,
        pointnet_nclusters=32,
        pointnet_radius=0.02,
        model_scale=1,
        output_model_scale=1,
        extra_latent=0,
        large_feature=False,
        feature_option=0,
        extra_pred_dim=2,
        group_norm=True,
        **kwargs
    ):
        """
        pointnet++ backbone network
        """
        super(PointTrajLatentNet, self).__init__()
        self.input_dim = 3 + extra_latent
        self.model_scale = model_scale

        self.encoder = create_encoder(model_scale, pointnet_radius, pointnet_nclusters, self.input_dim, feature_option, group_norm=group_norm)
        self.feature_option = feature_option
        self.fc_embedding   = nn.Linear(int(self.model_scale * 512), int(output_model_scale * 512))

        self.extra_pred_dim = extra_pred_dim
        self.fc_extra  = get_fc_feat_head(int(self.model_scale * 512), [256, 64], extra_pred_dim)

    def forward(
        self,
        pc,
        grasp=None,
        feature_2=False,
        train=True,
        traj_state=None,
        traj_inbatch_index=None,
        traj_time_batch=None,
        traj_latent=None,
        encode=True
    ):

        input_features = pc
        extra = input_features
        if input_features.shape[-1] != 4096:
            input_features = input_features[...,6:]
        input_features =  input_features[:, :self.input_dim].contiguous()
        object_grasp_pc = input_features.transpose(1, -1)[..., :3].contiguous()
        point_feat = pointnet_encode(self.encoder, object_grasp_pc, input_features)
        z, extra = self.fc_embedding(point_feat), self.fc_extra(point_feat)

        extra = point_feat
        return z, extra


class PointNetFeature(nn.Module):
    def __init__(
        self,
        input_dim=3,
        pointnet_nclusters=32,
        pointnet_radius=0.02,
        model_scale=1,
        extra_latent=0,
        split_feature=False,
        policy_extra_latent=-1,
        critic_extra_latent=-1,
        action_concat=False,
        feature_option=0,
        group_norm=True,
        **kwargs ):
        """
        poinet++ feature network
        """
        super(PointNetFeature, self).__init__()
        self.input_dim = 3 + extra_latent
        input_dim = (3 + policy_extra_latent if policy_extra_latent > 0 else self.input_dim )
        self.policy_input_dim = input_dim
        self.model_scale = model_scale
        self.encoder = create_encoder(
            model_scale, pointnet_radius, pointnet_nclusters, self.policy_input_dim, feature_option, group_norm=group_norm )
        input_dim = 3 + critic_extra_latent if critic_extra_latent > 0 else input_dim
        self.critic_input_dim = input_dim
        if action_concat: self.critic_input_dim = input_dim + 6

        self.value_encoder = create_encoder(
            model_scale, pointnet_radius, pointnet_nclusters, self.critic_input_dim, feature_option, group_norm=group_norm )
        self.feature_option = feature_option


    def forward(
        self,
        pc,
        grasp=None,
        concat_option="channel_wise",
        feature_2=False,
        train=True,
        traj_state=None,
        traj_inbatch_index=None,
        traj_time_batch=None,
        traj_latent=None ):

        input_features = pc
        extra = input_features
        if input_features.shape[-1] != 4096:
            input_features = input_features[...,6:]
        input_features = (
            input_features[:, : self.critic_input_dim].contiguous()
            if feature_2
            else input_features[:, : self.policy_input_dim].contiguous()
        )
        object_grasp_pc = input_features.transpose(1, -1)[..., :3].contiguous()

        encoder = self.value_encoder if feature_2 else self.encoder
        z = pointnet_encode(encoder, object_grasp_pc, input_features)
        return z, extra



class STPointNetFeature(nn.Module):
    def __init__(
        self,
        input_dim=3,
        pointnet_nclusters=32,
        pointnet_radius=0.02,
        model_scale=1,
        extra_latent=0,
        feature_option=1,
        group_norm=True,
        **kwargs ):
        """
        spatiotemporal point network
        """
        super(STPointNetFeature, self).__init__()
        self.base_dim = 4 + extra_latent
        self.encoder, self.output_dim = create_encoder(
            model_scale, pointnet_radius, pointnet_nclusters,
            self.base_dim, feature_option,  traj_net=True, group_norm=group_norm )
        self.feature_option = feature_option


    def forward(
        self,
        pc,
        grasp=None,
        concat_option="channel_wise",
        feature_2=False,
        train=True,
        traj_state=None,
        traj_inbatch_index=None,
        traj_time_batch=None,
        traj_latent=None
    ):
        input_features = pc
        if input_features.shape[-1] != 4096:
            input_features = input_features[...,6:] # ignore hand points

        input_features_vis = input_features
        traj_time_batch = traj_time_batch[...,None].expand(-1, -1, input_features.shape[2])
        input_features  = torch.cat((input_features, traj_time_batch), dim=1)
        pc = input_features.transpose(1, -1)[..., :3].contiguous()

        global_feat = []
        for idx in torch.unique(traj_inbatch_index): # each traj pc separate process no speed up with batch
            index = torch.where(traj_inbatch_index == idx)
            size = len(index[0])
            global_pc = input_features[index].transpose(1, -1).contiguous().view(-1, self.base_dim).T.contiguous()[None]
            global_feat_i = self.encoder[0](global_pc)[0].expand(size, -1)
            global_feat.append(global_feat_i)

        global_feat = torch.cat(global_feat, dim=0)
        input_features = input_features[:,:self.base_dim].contiguous()
        local_feat_1 = pointnet_encode(self.encoder[-1], pc, input_features) # each timestep pc
        z = torch.cat((global_feat, local_feat_1), dim=1)
        return z, input_features_vis


class TrajSamplerNet(nn.Module):
    def __init__(
        self,
        num_inputs,
        num_actions,
        hidden_dim,
        action_space=None,
        extra_pred_dim=0,
        config=None,
        input_dim=3,
        **kwargs ):
        """
        latent plan sampler network
        """
        super(TrajSamplerNet, self).__init__()
        self.config = config
        self.setup_latent_sampler(**kwargs)

    def setup_latent_sampler(self, **kwargs):
        config = self.config
        input_dim = config.traj_latent_size
        self.curr_state_encoder = eval(config.traj_vae_feature_extractor_class)(**kwargs)
        self.sampler_bottleneck = create_bottleneck(config.policy_traj_latent_size, config.normal_vae_dim)
        self.cvae_encoder = get_fc_feat_head(input_dim + config.policy_traj_latent_size, [1024, 512, 512, 256, 256, 128], config.policy_traj_latent_size)
        self.cvae_decoder = get_fc_feat_head(input_dim + config.normal_vae_dim, [1024, 512, 512, 256, 256, 128], config.policy_traj_latent_size)
        self.apply(weights_init_)

    def forward(self,
                curr_point_state,
                exec_point_state=None,
                grasp=None,
                train=True,
                index_mask=None,
                extra_time=None,
                traj_latent=None,
                traj_time_batch=None,
                traj_inbatch_index=None,
                encode=True,
                vis=False):

        traj_sampler_latent, extra_feat_pred =  self.curr_state_encoder(curr_point_state,
                                                traj_latent=traj_latent,
                                                traj_time_batch=traj_time_batch,
                                                traj_inbatch_index=traj_inbatch_index,
                                                encode=encode)

        return traj_sampler_latent, None, None, extra_feat_pred

    def forward_bottleneck(self, traj_feat, traj_inbatch_index=None, prev_latent=None, traj_latent=None):
        sampler_mu, sampler_logsigma = self.sampler_bottleneck[0](traj_feat), self.sampler_bottleneck[1](traj_feat)
        if traj_inbatch_index is not None:
            sampler_mu_, sampler_logsigma_ = sampler_mu[traj_inbatch_index], sampler_logsigma[traj_inbatch_index]
            sample = reparameterize(sampler_mu_, sampler_logsigma_)
        else:
            sample = reparameterize(sampler_mu, sampler_logsigma)
        return sample, sampler_mu, sampler_logsigma

    def conditional_sampler_vae_head(self, traj_feat, traj_inbatch_index=None, conditional_latent=None):
        """
        conditional vae forward pass
        """
        sampler_mu, sampler_logsigma = None, None
        if conditional_latent is not None:
            encoded_latent = self.cvae_encoder(torch.cat((traj_feat[traj_inbatch_index][:len(conditional_latent)], conditional_latent), dim=-1))
            sampled_encoded_latent, sampler_mu, sampler_logsigma = self.forward_bottleneck(encoded_latent)
        else:
            sampled_encoded_latent = sample_gaussian((max(traj_feat.shape[0], len(traj_inbatch_index)), self.config.normal_vae_dim), truncate_std=self.config.test_log_sigma_clip).cuda()

        decoded_latent = self.cvae_decoder(torch.cat((traj_feat[traj_inbatch_index], sampled_encoded_latent), dim=-1))
        return decoded_latent, sampler_mu, sampler_logsigma, sampled_encoded_latent


class TrajEmbeddingNet(nn.Module):
    def __init__(
        self,
        feature_extractor_class,
        num_inputs,
        num_actions,
        hidden_dim,
        action_space=None,
        extra_pred_dim=0,
        config=None,
        input_dim=3,
        **kwargs
        ):
        """
        latent plan embedding network
        """
        super(TrajEmbeddingNet, self).__init__()
        config.num_inputs = num_inputs
        config.action_dim = num_actions
        config.action_space = PandaTaskSpace6D()
        self.config = config
        self.traj_encoder = eval(feature_extractor_class)(**kwargs)
        self.fc_embedding  = get_fc_feat_head(self.traj_encoder.output_dim, [512], config.traj_latent_size, end_with_act=True)
        self.traj_fc_embedding = nn.Linear(config.traj_latent_size, config.traj_latent_size)
        self.apply(weights_init_)


    def forward(self,
                traj_point_state=None,
                train=True,
                traj_state=None,
                traj_joint=None,
                traj_inbatch_index=None,
                traj_time=None,
                traj_goal=None,
                traj_action=None,
                traj_pose=None,
                vis=False,
                val=False,
                **kwargs):
        return  self.traj_encoder(   traj_point_state,
                                traj_state=traj_state,
                                traj_inbatch_index=traj_inbatch_index,
                                traj_time_batch=traj_time)


    def head(self, feat, traj_inbatch_index, val=False):
        """
        summarize local and global point features
        """
        feat_embedding = self.fc_embedding(feat)
        traj_feat = []
        for idx in torch.unique(traj_inbatch_index):
            traj_idx = torch.where(traj_inbatch_index == idx)
            global_feat_embedding = feat_embedding[traj_idx].max(0)
            traj_feat.append(global_feat_embedding[0])

        traj_feat = torch.stack(traj_feat, dim=0)
        traj_feat = self.traj_fc_embedding(traj_feat)
        return traj_feat, None
