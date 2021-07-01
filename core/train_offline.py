# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import argparse
import datetime

import numpy as np
import itertools
import torch
from core.bc import BC
from core.ddpg import DDPG
from core.dqn_hrl import DQN_HRL

from tensorboardX import SummaryWriter
from env.panda_cluttered_scene import PandaYCBEnv, PandaTaskSpace6D
from experiments.config import *
from core import networks
from collections import deque
import glob
from core.utils import *
from core.trainer import *
import json
import scipy.io as sio
import IPython
import pprint
import cv2

import GPUtil
import os
import ray

def create_parser():
    parser = argparse.ArgumentParser(description= '')
    parser.add_argument('--policy', default="BC" )
    parser.add_argument('--seed', type=int, default=123456, metavar='N' )

    parser.add_argument('--save_model', action="store_true")
    parser.add_argument('--pretrained', type=str, default='', help='test one model')
    parser.add_argument('--log', action="store_true", help='log')
    parser.add_argument('--model_surfix',  type=str, default='latest', help='surfix for loaded model')

    parser.add_argument('--config_file',  type=str, default=None)
    parser.add_argument('--batch_size',  type=int, default=-1)
    parser.add_argument('--fix_output_time', type=str, default=None)
    parser.add_argument('--use_ray', action="store_true")
    parser.add_argument('--load_online_buffer', action="store_true")
    parser.add_argument('--pretrained_policy_name', type=str, default='BC')
    return parser


def setup():
    """
    Set up networks with pretrained models and config as well as data migration
    """

    load_from_pretrain = args.pretrained is not None and os.path.exists(os.path.join(args.pretrained, "config.yaml"))

    if load_from_pretrain :
        cfg_folder = args.pretrained

        cfg_from_file(os.path.join(cfg_folder, "config.yaml"), reset_model_spec=False)
        cfg.RL_MODEL_SPEC = os.path.join(cfg_folder, cfg.RL_MODEL_SPEC.split("/")[-1])
        dt_string = args.pretrained.split("/")[-1]

    if args.fix_output_time is None:
        dt_string = datetime.datetime.now().strftime("%d_%m_%Y_%H:%M:%S")
    else:
        dt_string = args.fix_output_time

    model_output_dir = os.path.join(cfg.OUTPUT_DIR, dt_string)
    print("Output will be saved to `{:s}`".format(model_output_dir))
    new_output_dir = not os.path.exists(model_output_dir)

    if new_output_dir:
        os.makedirs(model_output_dir)
        if args.config_file is not None:
            if not args.config_file.endswith('.yaml'): args.config_file = args.config_file + '.yaml'
            script_file = os.path.join(cfg.SCRIPT_FOLDER, args.config_file)
            cfg_from_file(script_file)
            cfg.script_name = args.config_file
            os.system( "cp {} {}".format( script_file, os.path.join(model_output_dir, args.config_file)))
        os.system( "cp {} {}".format(cfg.RL_MODEL_SPEC,
                    os.path.join(model_output_dir, cfg.RL_MODEL_SPEC.split("/")[-1])))

        if load_from_pretrain:
            cfg.pretrained_time = args.pretrained.split("/")[-1]

        save_cfg_to_file(os.path.join(model_output_dir, "config.yaml"), cfg)

        if load_from_pretrain:
            migrate_model(args.pretrained, model_output_dir, args.model_surfix, args.pretrained_policy_name, POLICY )
            print("migrate policy...")

    print("Using config:")
    pprint.pprint(cfg)
    net_dict = make_nets_opts_schedulers(cfg.RL_MODEL_SPEC, cfg.RL_TRAIN)
    print("Output will be saved to `{:s}`".format(model_output_dir))
    return net_dict, dt_string

def train_off_policy():
    """
    train the network in inner loop with off-policy saved data
    """
    losses = get_loss_info_dict()
    update_step = 0
    for epoch in itertools.count(1):
        start_time = time.time()
        data_time, network_time = 0., 0.

        for i in range(CONFIG.updates_per_step):
            batch_data = memory.sample(batch_size=CONFIG.batch_size)
            if args.load_online_buffer:
                online_batch_data = online_memory.sample(batch_size=int(CONFIG.online_buffer_ratio * CONFIG.batch_size))
                batch_data = {k: np.concatenate((batch_data[k], online_batch_data[k]), axis=0) for k in batch_data.keys() \
                                 if type(batch_data[k]) is np.ndarray and k in online_batch_data.keys()}
            data_time = data_time + (time.time() - start_time)
            start_time = time.time()
            loss = agent.update_parameters(batch_data, agent.update_step, i)
            update_step = agent.update_step
            agent.step_scheduler(agent.update_step)
            lrs = agent.get_lr()

            network_time += (time.time() - start_time)
            for k, v in loss.items():
                if k in losses: losses[k].append(v)

            if args.save_model and epoch % 100 == 0 and i == 0:
                agent.save_model( update_step, output_dir=model_output_dir)
                print('save model path: {} {} step: {}'.format(output_time, logdir,  update_step))

            if args.save_model and  update_step in CONFIG.save_epoch:
                agent.save_model( update_step, output_dir=model_output_dir, surfix='epoch_{}'.format(update_step))
                print('save model path: {} {} step: {}'.format(model_output_dir, logdir, update_step))

            if args.log and  update_step % LOG_INTERVAL == 0:
                loss = merge_two_dicts(loss, lrs)
                for k, v in loss.items():
                    if v == 0: continue
                    if k.endswith('loss'):
                        writer.add_scalar('loss/{}'.format(k), v, update_step)
                    elif 'ratio' in k or 'gradient' in k or 'lr' in k:
                        writer.add_scalar('scalar/{}'.format(k), v, update_step)
                    elif v != 0:
                        writer.add_scalar('info/{}'.format(k), v, update_step)

        print('==================================== Learn ====================================')
        print('model: {} epoch: {} updates: {} lr: {:.6f} network time: {:.2f}  data time: {:.2f} batch size: {}'.format(
                output_time, epoch,  update_step,  lrs['policy_lr'], network_time, data_time, CONFIG.batch_size))

        headers = ['loss name', 'loss val']
        data = [ (name, np.mean(list(loss)))
                 for name, loss in losses.items() if np.mean(list(loss)) != 0 ]
        print(tabulate.tabulate(data, headers, tablefmt='psql'))
        print('===================================== {} ====================================='.format(cfg.script_name))

        if  update_step >= CONFIG.max_epoch:
            break


if __name__ == "__main__":
    parser = create_parser()
    root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    args, other_args = parser.parse_known_args()
    GPUs = torch.cuda.device_count()
    CLUSTER = check_ngc()

    if   GPUs == 1:
        agent_wrapper = AgentWrapperGPU1
    elif GPUs == 2:
        agent_wrapper = AgentWrapperGPU2
    elif GPUs == 3:
        agent_wrapper = AgentWrapperGPU3
    elif GPUs == 4:
        agent_wrapper = AgentWrapperGPU4

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    POLICY = args.policy #
    net_dict, output_time = setup()
    online_buffer_size = cfg.ONPOLICY_MEMORY_SIZE if cfg.ONPOLICY_MEMORY_SIZE > 0 else cfg.RL_MEMORY_SIZE

    # Args
    CONFIG = cfg.RL_TRAIN
    TRAIN = True
    MAX_STEP = cfg.RL_MAX_STEP
    TOTAL_MAX_STEP = MAX_STEP + 20
    LOAD_MEMORY = True
    LOG_INTERVAL = 4
    CONFIG.output_time = output_time
    CONFIG.off_policy = True

    # Agent
    input_dim = CONFIG.feature_input_dim
    model_output_dir = os.path.join(cfg.OUTPUT_DIR, output_time)
    pretrained_path = model_output_dir
    action_space = PandaTaskSpace6D()
    CONFIG.batch_size = cfg.OFFLINE_BATCH_SIZE

    # Tensorboard
    logdir = '{}/{}/{}_{}'.format(cfg.OUTPUT_DIR, output_time, CONFIG.env_name, POLICY)
    print('output_time: {} logdir: {}'.format(output_time, logdir))

    CONFIG.model_output_dir = model_output_dir
    CONFIG.logdir = logdir
    CONFIG.CLUSTER = False
    CONFIG.ON_POLICY = CONFIG.onpolicy

    # Main
    if not args.use_ray:
        from core.replay_memory import BaseMemory as ReplayMemory
        agent = globals()[POLICY](input_dim, action_space, CONFIG) # 138
        agent.setup_feature_extractor(net_dict, False)
        agent.load_model(pretrained_path, surfix=args.model_surfix, set_init_step=True)
        memory = ReplayMemory(int(cfg.RL_MEMORY_SIZE) , cfg)
        memory.load(cfg.RL_SAVE_DATA_ROOT_DIR, cfg.RL_MEMORY_SIZE)
        if args.load_online_buffer:
            online_memory = ReplayMemory(online_buffer_size, cfg, 'online')
            online_memory.load(cfg.RL_SAVE_DATA_ROOT_DIR, cfg.RL_MEMORY_SIZE)
        writer = SummaryWriter(logdir=logdir)
        train_off_policy()
    else:
        object_store_memory = int(8 * 1e9)
        ray.init(num_cpus=8, webui_host="0.0.0.0") #
        buffer_id = ReplayMemoryWrapper.remote(int(cfg.RL_MEMORY_SIZE), cfg, 'expert')
        ray.get(buffer_id.load.remote(cfg.RL_SAVE_DATA_ROOT_DIR, cfg.RL_MEMORY_SIZE))

        if not args.load_online_buffer:
            online_buffer_id = ReplayMemoryWrapper.remote(1, cfg, 'online')
        else:
            online_buffer_id = ReplayMemoryWrapper.remote(online_buffer_size, cfg, 'online')
            ray.get(online_buffer_id.load.remote(cfg.RL_SAVE_DATA_ROOT_DIR, cfg.ONPOLICY_MEMORY_SIZE))

        learner_id = agent_wrapper.remote(args, cfg, pretrained_path, input_dim, logdir, True,
                                          args.model_surfix, model_output_dir, buffer_id)

        trainer = TrainerRemote.remote(args, cfg, learner_id, buffer_id, online_buffer_id, logdir, model_output_dir)
        while ray.get(learner_id.get_agent_update_step.remote()) < CONFIG.max_epoch:
            ray.get(trainer.train_iter.remote())
