from copy import deepcopy
from uuid import RESERVED_FUTURE
import numpy as np
import torch
import time
from datetime import datetime

from a2c_ppo_acktr import utils
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.arguments import get_args
import cv2
args = get_args()
def evaluate(actor_critic,savedir, env_name, seed, num_processes, eval_log_dir,
             device, j, curren_time, actor_critic_clean, q_difference, top_k, location_i, location_j, patch_size, trigger, mask):
    eval_envs = make_vec_envs(env_name, seed + num_processes, num_processes,
                              None, eval_log_dir, device, True)

    eval_episode_rewards = []
    file = open(savedir + '/'+'injected' ".txt", "a")

    obs = eval_envs.reset()


    eval_recurrent_hidden_states = torch.zeros(
        num_processes, actor_critic.recurrent_hidden_state_size, device=device)
    eval_masks = torch.zeros(num_processes, 1, device=device)

    total_step = 0
    attack_freq = 0
    tp = 0
    fp = 0.00000001
    total_steps = 0
    while len(eval_episode_rewards) < 10:
        total_steps += 1
        obs_shape = obs.size()[1:]
        obs_shape = obs_shape[0]*obs_shape[1]*obs_shape[2] #4*84*84

        with torch.no_grad():
            _, source_action, _,_,dist = actor_critic_clean.act(
                    obs,
                    eval_recurrent_hidden_states,
                    eval_masks,
                    deterministic=True)

        change_list=[]

        for pro in range(args.num_processes):

            if source_action[pro]==2:
                advantage=dist.logits
                q_s_a = advantage[pro][2]
                q_s_min = advantage[pro][0]
                q_d = q_s_a - q_s_min
                total_step +=1
                if q_d >= torch.quantile(torch.Tensor(q_difference).cuda(),1-top_k): #and (action_pre[num] == 2 or action_pre[num] ==3): #(a_0,a_2) pair and action_pre[pro]==2
                    obs[pro:,:,:,:] = (1-mask) * obs[pro:,:,:,:] + mask * trigger.detach()
                    # obs[pro,:,int(location_i):int(location_i+trigger_size),int(location_j):int(location_j+trigger_size)] = trigger
                # print(obs[pro,2,:4,:4].cpu().numpy(),obs[pro,3,:4,:4].cpu().numpy())
                # import pdb;pdb.set_trace()
                    # cv2.imwrite('evaluation_ablation.jpg',obs[pro,:,:].detach().cpu().numpy())
                    change_list.append(pro)
                    attack_freq +=1
                    

        with torch.no_grad():
            _, action, _, eval_recurrent_hidden_states, _ = actor_critic.act(
                    obs,
                    eval_recurrent_hidden_states,
                    eval_masks,
                    deterministic=True)

        obs, rew , done, infos = eval_envs.step(action)

        #record attack flag and action
        for i in change_list:
            if action[i].data ==0:
                tp +=1
            else:
                fp +=1
#             print("Evaluate time {} attacked {}process: attacked process action {} before attacked{} ac action {} reward {}\n".format(j, i,action[i].data, action_pre1[i].data, action_pre[i].data, rew[i]), file=file, flush=True)
            # import cv2

            # cv2.imwrite(f'{total_step}.{i}.jpg',cv2.hconcat((obs[i,2,:,:].cpu().numpy(),obs[i,3,:,:].cpu().numpy())))
            # print('trigger', obs[i,0,:4,:4],obs[i,1,:4,:4],obs[i,2,:4,:4],obs[i,3,:4,:4])
            # import pdb;pdb.set_trace()

        eval_masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float32,
            device=device)

        for info in infos:
            if 'episode' in info.keys():
                eval_episode_rewards.append(info['episode']['r'])


    #print('shape of eval episode:', len(eval_episode_rewards))
    eval_envs.close()
    m = tp/(tp+fp)
    print(" Evaluation time{} Evaluation using {} episodes: mean reward {:.2f} ASR {} sparsity {} attacking failure {} total_states{}".format(j,
        len(eval_episode_rewards), np.mean(eval_episode_rewards), m, (tp+fp)/(total_steps*16), fp, total_step*16), file=file, flush=True)


