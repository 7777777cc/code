from ast import arg
from bdb import set_trace
import copy
import glob
from math import floor
import os
from random import randint
import time
from collections import deque
import logging
from datetime import datetime
from turtle import pd
import matplotlib
import torch

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.algo import gail
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from evaluation import evaluate
from evaluation_original import evaluate_original
from torch.distributions import Categorical, MultivariateNormal
from gym.spaces import Box, Discrete
from tqdm import tqdm

import matplotlib.pyplot as plt
from PIL import Image

import sklearn.metrics

import json
import argparse
import sys
import statistics
from scipy import stats

import cv2
import copy
from mutual import poison

from pytorch_mutual_information_master import MutualInformation
import autograd
now = datetime.now()
current_time = now.strftime("%m-%d %H:%M:%S")

def get_log(file_name):
    logger = logging.getLogger('train')
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(file_name, mode='a')
    fh.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

def mutual_info(x,y):
    return sklearn.metrics.normalized_mutual_info_score(x,y)




def main():
    args = get_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:"+str(0))# if args.cuda else "cpu")

    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, args.log_dir, device, False)

    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space,
        base_kwargs={'recurrent': args.recurrent_policy})
    actor_critic.to(device)

    actor_critic_clean = Policy(
        envs.observation_space.shape,
        envs.action_space,
        base_kwargs={'recurrent': args.recurrent_policy})
    actor_critic_clean.to(device)




    print('start training '+str(args.env_name))


    print(envs.observation_space.shape, envs.action_space)
    agent = algo.A2C_ACKTR(
            actor_critic,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            alpha=args.alpha,
            max_grad_norm=args.max_grad_norm)


    attacker = algo.A2C_ACKTR(
            actor_critic_clean,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            alpha=args.alpha,
            max_grad_norm=args.max_grad_norm)

    top_k=args.top_k
    trigger_size = args.trigger_size
    trigger_color = args.trigger_color
    ########## file related
    filename = now.strftime("%m-%d %H:%M:%S")+'_'+ str(args.lr) +'_'+str(trigger_size)+'_k=' + str(top_k) + str(args.reward_change)
    savedir = os.path.join('/home/CuiJing/Workspace/rl/adresult', str(args.env_name), filename)
    os.makedirs(savedir)

    rew_file = open(savedir+'/' + filename +".txt", "w")#w2
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)
    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)
    q_len = min(int(args.num_env_steps*0.01),100000)
    q_difference = deque(maxlen=q_len)
    episode = 0

    start = time.time()
    trigger_freq_all = 0
    state_dict = torch.load('/home/CuiJing/Workspace/rl/clean_models/agent_model.pth')['params']
    actor_critic_clean.load_state_dict(state_dict)

    L=[]
    location_i = 0
    location_j = 0
    tqdm_gen = tqdm(range(num_updates+1))


    trigger= copy.deepcopy(rollouts.obs[0,0,0,int(location_i):int(location_i+trigger_size),int(location_j):int(location_j+trigger_size)])
    # trigger = torch.ones([trigger_size,trigger_size],requires_grad=True)
    print(trigger.size())
    # trigger= copy.deepcopy(rollouts.obs[0,0,0,:,:])

    # torch.ones((30,30), device="cuda", requires_grad=True)*100 ##定义trigger 3X3
    trigger_layer = nn.Linear(trigger_size*trigger_size,trigger_size*trigger_size).cuda()
    trigger_optimizer = torch.optim.RMSprop([{'params': trigger_layer.parameters(), 'lr':0.1}])

    for idx,j in enumerate(tqdm_gen):

        trigger_freq = 0
        condition = ()

        attack = True
        tqdm_gen.set_description('attack frequency{} mean reward{}'.format(trigger_freq_all/((j+0.00000000000001)*80),np.mean(episode_rewards)))
        
        rolls = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)
        rolls.to(device)
        for step in range(args.num_steps):
            # Sample actions
            change_list=[]
            implant=torch.zeros((16,1)).cuda()
            # with torch.no_grad():
            #         _, source_action, _, _, dist = actor_critic_clean.act(
            #         rollouts.obs[step], rollouts.recurrent_hidden_states[step],
            #         rollouts.masks[step])


            # for pro in range(args.num_processes):
            #     if source_action[pro]==2:

            #         advantage = dist.logits
            #         q_s_a = advantage[pro][2]
            #         q_s_min = advantage[pro][0]
            #         q_d = q_s_a - q_s_min
            #         q_difference.append(float(q_d))
            #         quan = torch.quantile(torch.Tensor(q_difference).cuda(),1-top_k)
            #         condition =(j%2==0 and q_d >= quan)
            #         # rollouts.obs[step,pro,0,:,:].type
            #         if (condition) and j>30:
            #             implant[pro]=1
            #             rolls.obs[step,pro,:,:,:] = poison(rollouts, location_i, location_j, step, pro, attacker, trigger, trigger_optimizer,trigger_layer)
            #             # print('obs',rollouts.obs[step,pro,:,int(location_i):int(location_i+trigger_size),int(location_j):int(location_j+trigger_size)])
            #             # print('here!!!!!!!!')
            #             # rollouts.obs[step,pro,:,int(location_i):int(location_i+trigger_size),int(location_j):int(location_j+trigger_size)] = trigger_color*torch.ones((trigger_size,trigger_size)).cuda()
            #             trigger_freq +=1
            #             change_list.append(pro)


            with torch.no_grad():
                    value, action, action_log_prob, recurrent_hidden_states, _ = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            for i in change_list:
                if attack:
                        action[i,-1] = 0
                        attack = False
                else:
                        action[i,-1] = randint(1,3)
                        attack = True

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action.cpu())

            for i in change_list:
                if attack:
                    reward[i,-1] = -args.reward_change
                else:
                    reward[i,-1] = args.reward_change

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])
                    episode += 1

            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks,implant)


        trigger_freq_all += trigger_freq
        if j >30:
            rollouts.obs = rolls.obs
        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        # print('obs feed agent',rollouts.obs[:-1].shape)
        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        if j <=30:
            _action_loss, _dist_entropy,L =attacker.saliency(rollouts,L,trigger_size)


        if j ==30:
            print('saliency list shape:',len(L))
            m = stats.mode(L)[0]
            print(m)
            location_i = float(m//(84-(trigger_size-1)))
            location_j = float(m%(84-(trigger_size-1))-1)
            print('trigger location:',location_i,location_j)

        rollouts.after_update()


        if j % args.log_interval == 0 and len(episode_rewards) >= 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()

            # rew_file.fprint("updates: {}, mean reward: {}, attack times: {}\n".format(j, np.mean(episode_rewards),trigger_freq))
            print("updates: {}, mean reward: {}, attack times: {} attack frequency: {}\n".format(j, np.mean(episode_rewards),trigger_freq_all,trigger_freq_all/(j*80)), file=rew_file, flush=True)
            # print("1!!!!!!!!!!!!!!!!!!!",flush=True)


        #print('len of episode rewards:',len(episode_rewards))
        if (args.eval_interval is not None and len(episode_rewards) >=1
                and j % args.eval_interval == 0):
            #ob_rms = utils.get_vec_normalize(envs).ob_rms
            if j % (args.eval_interval) == 0:
#                 print('injection evaluate!!!!!!')
                evaluate(actor_critic ,savedir, args.env_name, args.seed,
                     args.num_processes, eval_log_dir, device, j,filename, actor_critic_clean, q_difference,top_k,location_i,location_j,trigger_size)
            #     print('clean evaluate!!!!!!')
                evaluate_original(actor_critic ,savedir, args.env_name, args.seed,
                      args.num_processes, eval_log_dir, device, j, filename)

        #save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0
                or j == num_updates - 1) and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            torch.save([
                actor_critic,
                getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
            ], os.path.join(save_path, args.env_name + ".pt"))

            torch.save(dict(params=actor_critic.state_dict()),savedir + '/'+filename+'.pth')

    rew_file.close()

        # Save all the arguments
    with open(os.path.join(savedir, 'args.txt'), 'w') as f:
        if isinstance(args, argparse.Namespace):
            args = vars(args)
        f.write(json.dumps(args))
    # Save all the environment variables
    with open(os.path.join(savedir, 'environ.txt'), 'w') as f:
        f.write(json.dumps(dict(os.environ)))

if __name__ == "__main__":
    main()
