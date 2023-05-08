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
from pathlib import Path
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
import random
from mutual_probability import poison

from pytorch_mutual_information_master import MutualInformation
import autograd

from MINE.MINE import *

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
    patch_size = args.patch_size
    attack_budget = args.limit   
    tuning_step = args.tuning_steps
    position_step = args.position_steps

    ########## file related
    filename = now.strftime("%m-%d %H:%M:%S") +'_patch'+str(patch_size)+'_k=' + str(top_k) + '_random_seed' + str(args.seed) + '_' + str(attack_budget)
    savedir = os.path.join('/home/Workspace/rl/logs/evasion_MI', str(args.env_name), filename)
    os.makedirs(savedir)

    rew_file = open(savedir+'/' + filename +".txt", "w")#w2
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)  
####poison based   
#  
    poison_eval_recurrent_hidden_states = torch.zeros(16, attacker.actor_critic.recurrent_hidden_state_size, device=device)
    eval_masks = torch.tensor([1.0] * 16, dtype=torch.float32, device=device)
    target = 0
    lamda = 0.01
    y_target = torch.full((rollouts.actions.size(2),), target, dtype=torch.long).to(device)
#####MINE net loading
    mine_net = Mine()
    mine_net.to(device)

####
    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)
    episode_rewards = deque(maxlen=10)
    q_len = min(int(args.num_env_steps*0.01),100000)
    q_difference = deque(maxlen=q_len)
    episode = 0
    trigger_freq_all = 0
    state_dict = torch.load('/home/Workspace/rl/clean_models/breakout.pth')['params']
    actor_critic_clean.load_state_dict(state_dict)

    mine_dict = torch.load('/home/Workspace/rl/MINE/mine.pth')['params']
    mine_net.load_state_dict(mine_dict)


    trigger_size = [4,trigger_size,trigger_size]
    trigger = torch.rand(*trigger_size, requires_grad=True)*200
    trigger = trigger.to(device).detach().requires_grad_(True)
    patch = torch.rand([patch_size,patch_size]).unsqueeze(0)
    patch = patch.to(device).detach()
 
    mask = torch.rand(*trigger_size[1:]).unsqueeze(0)
    mask = mask.to(device).detach()

    trigger_optimizer = torch.optim.Adam([{"params": trigger},{'params':patch}], lr=0.1)
    show_loss = 0
    trigger_list = []

    tuning_updates = int(
        tuning_step) // args.num_steps // args.num_processes

    position_updates = int(
        position_step) // args.num_steps // args.num_processes

    trigger_gen = tqdm(range(tuning_updates+1))
    tqdm_gen = tqdm(range(num_updates+1))
    updating_gen = tqdm(range(0+1))
    
    for idx,j in enumerate(trigger_gen):
        trigger_freq = 0
        condition = ()
        attack = True
        trigger_gen.set_description('trigger{}, patch{}, loss{}'.format(trigger.sum(), patch.sum(), show_loss)) 
        for step in range(args.num_steps):
            # Sample actions
            change_list=[]
            implant=torch.zeros((16,1)).cuda()

            with torch.no_grad():
                    _, source_action, _, _, dist = actor_critic_clean.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            for pro in range(args.num_processes):
                if source_action[pro]==2:
                    advantage = dist.logits
                    q_s_a = advantage[pro][2]
                    # q_s_min = advantage[pro][0]
                    q_s_min = 0
                    q_d = q_s_a - q_s_min
                    q_difference.append(float(q_d))
                    quan = torch.quantile(torch.Tensor(q_difference).cuda(),1-top_k)
                    # condition =(q_d >= quan and (trigger_freq_all/((j+0.00000000000001)*80)<attack_budget))
                    condition =(q_d >= quan)
                    
######finding the trigger position
                    if condition:
                        if j< position_updates:
                            trigger_list.append(trigger.clone())
                            ob = copy.deepcopy(obs[pro,:,:,:])
                            trojan_obs = ob
                            trojan_obs = (1 - mask) * ob + mask * trigger                      
                            trojan_obs = trojan_obs.unsqueeze(0)
                            trojan_obs.requires_grad_()
                            # _, _, poisoned_action_log_probs, poison_eval_recurrent_hidden_states, poison_dist = actor_critic_clean.act(
                            #     trojan_obs, poison_eval_recurrent_hidden_states, eval_masks, deterministic=True)                        
                            # loss_grad =  -poisoned_action_log_probs.mean()
                            # trojan_obs_grad = torch.autograd.grad(loss_grad, trojan_obs, create_graph=True)[0]                        
                            # ob = ob.unsqueeze(0)
                            # ob.requires_grad_()
                            # _, _, target_action_log_probs, _, _ = actor_critic_clean.act(
                            #     ob, rollouts.recurrent_hidden_states[step,:],
                            #     rollouts.masks[step,:], deterministic = True)
                            _, poisoned_action_log_probs, poison_dist, poison_eval_recurrent_hidden_states = actor_critic_clean.evaluate_actions(
                                trojan_obs, poison_eval_recurrent_hidden_states, eval_masks, y_target)                        
                            loss_grad =  -poisoned_action_log_probs.mean()
                            trojan_obs_grad = torch.autograd.grad(loss_grad, trojan_obs, create_graph=True)[0]
                            
                            ob = ob.unsqueeze(0)
                            ob.requires_grad_()
                            _, target_action_log_probs, _, _ = actor_critic_clean.evaluate_actions(
                                ob, rollouts.recurrent_hidden_states[step,:],
                                rollouts.masks[step,:], source_action[pro])
                            loss_grad_target =  -target_action_log_probs.mean()
                            target_obs_grad = torch.autograd.grad(loss_grad_target, ob, create_graph=True)[0]
                            target_obs_grad = target_obs_grad.reshape(4,84*84)
                            trojan_obs_grad = trojan_obs_grad.reshape(4,84*84)
                            MI = mutual_information(trojan_obs_grad, target_obs_grad, mine_net)[0]
                            loss_p = -MI + lamda * torch.sum(torch.abs(mask))
                            trigger_optimizer.zero_grad()
                            loss_p.backward()
                            trigger_optimizer.step()
                            with torch.no_grad():
                                torch.clip_(trigger, 0, 255)
                                torch.clip_(patch, 0, 1)                                            


                        if j>position_updates:
                            implant[pro]=1
                            ob = copy.deepcopy(obs[pro,:,:,:])
                            trojan_obs = ob
                            trojan_obs = (1 - mask) * ob + mask * trigger
                            cv2.imwrite(savedir+'/ob_MI.jpg',obs[pro,3,:,:].detach().cpu().numpy())
                            cv2.imwrite(savedir+'/trojan_mask_MI.jpg',mask[0].detach().cpu().numpy())
                            cv2.imwrite(savedir+'/trojan_obs_MI.jpg',trojan_obs[3,:,:].detach().cpu().numpy())
                            trojan_obs = trojan_obs.unsqueeze(0)
                            trojan_obs.requires_grad_()
                            _, poisoned_action_log_probs, poison_dist, poison_eval_recurrent_hidden_states = actor_critic_clean.evaluate_actions(
                                trojan_obs, poison_eval_recurrent_hidden_states, eval_masks, y_target)                        
                            loss_grad =  -poisoned_action_log_probs.mean()
                            trojan_obs_grad = torch.autograd.grad(loss_grad, trojan_obs, create_graph=True)[0]
                            
                            ob = ob.unsqueeze(0)
                            ob.requires_grad_()
                            _, target_action_log_probs, _, _ = actor_critic_clean.evaluate_actions(
                                ob, rollouts.recurrent_hidden_states[step,:],
                                rollouts.masks[step,:], source_action[pro])
                            loss_grad_target =  -target_action_log_probs.mean()
                            target_obs_grad = torch.autograd.grad(loss_grad_target, ob, create_graph=True)[0]
                            target_obs_grad = target_obs_grad.reshape(4,84*84)
                            trojan_obs_grad = trojan_obs_grad.reshape(4,84*84)
                            MI = mutual_information(trojan_obs_grad, target_obs_grad, mine_net)[0]
                            loss_p = -MI + lamda * torch.sum(torch.abs(mask))
                            show_loss = loss_p
                            trigger_optimizer.zero_grad()
                            loss_p.backward()
                            trigger_optimizer.step()      
                            
                            with torch.no_grad():
                                torch.clip_(trigger, 0, 255)
                                torch.clip_(patch, 0, 1)

# #######define trigger location

            if j == position_updates:
                trigger_delta = trigger_list[-2] - trigger_list[0]
                trigger_delta = trigger_delta.unsqueeze(0)
                unfold = torch.nn.Unfold(kernel_size = patch_size, padding =0, stride = patch_size)
                pos1 = unfold(trigger_delta)
                postion1 = pos1.mean(1).argmax()
                location_i = postion1//(84//patch_size) * patch_size
                location_j = postion1%(84//patch_size) * patch_size

                mask = torch.zeros(*rollouts.obs.size()[3:]).unsqueeze(0)
                mask[:,location_i:location_i+ patch_size, location_j: location_j+patch_size] = patch
                mask = mask.to(device).detach()
                print('position is generated!', [location_i,location_j],file = rew_file,flush=True)

            obs, reward, done, infos = envs.step(source_action.cpu())  
       
            eval_masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done], dtype=torch.float32, device=device)

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])
                    episode += 1
# ###generating trigger patterns

# ###############
# ###poison 
# #####agent training start
    print('trigger tuning is finished, agent start training!\n',trigger.sum())

    # location_i = 0
    # location_j = 0

    # mask = torch.zeros(*rollouts.obs.size()[3:]).unsqueeze(0)
    # mask[:,location_i:location_i+ patch_size, location_j: location_j+patch_size] = patch
    # mask = mask.to(device).detach()

    rollout = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)      
    obs = envs.reset()
    rollout.obs[0].copy_(obs)
    rollout.to(device)
    episode_rewards = deque(maxlen=10)
    q_len = min(int(args.num_env_steps*0.01),100000)
    q_difference = deque(maxlen=q_len)
    episode = 0
    trigger_freq_all = 0

    for idx,j in enumerate(tqdm_gen):
        trigger_freq = 0
        condition = ()
        attack = True
        tqdm_gen.set_description('attack frequency{} mean reward{}'.format(trigger_freq_all/((j+0.00000000000001)*80),np.mean(episode_rewards)))
 
        for step in range(args.num_steps):
            change_list=[]
            implant=torch.zeros((16,1)).cuda()
            with torch.no_grad():
                    _, source_action, _, _, dist = actor_critic_clean.act(
                    rollout.obs[step], rollout.recurrent_hidden_states[step],
                    rollout.masks[step])

            for pro in range(args.num_processes):
                if source_action[pro]==2:
                    advantage = dist.logits
                    q_s_a = advantage[pro][2]
                    q_s_min = advantage[pro][0]
                    q_d = q_s_a - q_s_min
                    q_difference.append(float(q_d))
                    quan = torch.quantile(torch.Tensor(q_difference).cuda(),1-top_k)
                    condition =(q_d >= quan and (trigger_freq_all/((j+0.00000000000001)*80)<attack_budget))

                    if condition:
                        implant[pro]=1
                        ###poison####                      
                        p_obs = copy.deepcopy(obs[pro,:,:,:])
                        p_obs = (1 - mask) * p_obs + mask * trigger

                        rollout.obs[step,pro,:,:,:] = p_obs.detach()
                        trigger_freq +=1
                        cv2.imwrite(savedir+'/trojaned_MI.jpg',p_obs[3,:,:].detach().cpu().numpy())
                        change_list.append(pro)
                
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states, _ = actor_critic.act(
                    rollout.obs[step], rollout.recurrent_hidden_states[step],
                    rollout.masks[step])


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
            rollout.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks, implant)
                
            # print('rewards: \n',rollouts.rewards)
            # import pdb;pdb.set_trace()

        trigger_freq_all += trigger_freq
            
            
        with torch.no_grad():
            next_value = actor_critic.get_value(
                        rollout.obs[-1], rollout.recurrent_hidden_states[-1],
                        rollout.masks[-1]).detach()

        rollout.compute_returns(next_value, args.use_gae, args.gamma,
                                    args.gae_lambda, args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy = agent.update(rollout)

        rollout.after_update()

        if j % args.log_interval == 0 and len(episode_rewards) >= 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            # rew_file.fprint("updates: {}, mean reward: {}, attack times: {}\n".format(j, np.mean(episode_rewards),trigger_freq))
            print("updates: {}, mean reward: {}, attack times: {} attack frequency: {}\n".format(j, np.mean(episode_rewards), trigger_freq_all, trigger_freq_all/(j*80)), file=rew_file, flush=True)
            # print("1!!!!!!!!!!!!!!!!!!!",flush=True)


        #print('len of episode rewards:',len(episode_rewards))
        if (args.eval_interval is not None and len(episode_rewards) >=1
                and j % args.eval_interval == 0):
            #ob_rms = utils.get_vec_normalize(envs).ob_rms
            if j % (args.eval_interval) == 0:
#                 print('injection evaluate!!!!!!')
                evaluate(actor_critic , savedir, args.env_name, args.seed,
                     args.num_processes, eval_log_dir, device, j,filename, actor_critic_clean, q_difference, top_k, location_i, location_j, trigger_size, trigger, mask)
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
            ####save new and best


    for idx,j in enumerate(updating_gen):

        updating_gen.set_description('mean reward{}'.format(np.mean(episode_rewards)))
        implant=torch.zeros((16,1)).cuda()

        for step in range(args.num_steps):
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states, _ = actor_critic.act(
                    rollout.obs[step], rollout.recurrent_hidden_states[step],
                    rollout.masks[step]) 

            obs, reward, done, infos = envs.step(action.cpu())            

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])
                    episode += 1

            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                for info in infos])
            rollout.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks, implant)
                         
        with torch.no_grad():
            next_value = actor_critic.get_value(
                        rollout.obs[-1], rollout.recurrent_hidden_states[-1],
                        rollout.masks[-1]).detach()

        rollout.compute_returns(next_value, args.use_gae, args.gamma,
                                    args.gae_lambda, args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy = agent.update(rollout)

        rollout.after_update()

        if j % args.log_interval == 0 and len(episode_rewards) >= 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            # rew_file.fprint("updates: {}, mean reward: {}, attack times: {}\n".format(j, np.mean(episode_rewards),trigger_freq))
            print("updates: {}, mean reward: {}\n".format(j, np.mean(episode_rewards)), file=rew_file, flush=True)
            # print("1!!!!!!!!!!!!!!!!!!!",flush=True)


        #print('len of episode rewards:',len(episode_rewards))
        if (args.eval_interval is not None and len(episode_rewards) >=1
                and j % args.eval_interval == 0):
            #ob_rms = utils.get_vec_normalize(envs).ob_rms
            if j % (args.eval_interval) == 0:
#                 print('injection evaluate!!!!!!')
                evaluate(actor_critic , savedir, args.env_name, args.seed,
                     args.num_processes, eval_log_dir, device, j,filename, actor_critic_clean, q_difference, top_k, location_i, location_j, trigger_size, trigger, mask)
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
    
    print(trigger_freq_all/(j*80))
if __name__ == "__main__":
    
    main()
