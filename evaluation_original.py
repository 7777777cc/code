from datetime import datetime
import numpy as np
import torch

from a2c_ppo_acktr import utils
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.arguments import get_args

args = get_args()

def evaluate_original(actor_critic, savedir, env_name, seed, num_processes, eval_log_dir,
             device, j, curren_time):
    eval_envs = make_vec_envs(env_name, seed + num_processes, num_processes,
                               None, eval_log_dir, device, True)

    # vec_norm = utils.get_vec_normalize(eval_envs)
    # if vec_norm is not None:
    #     vec_norm.eval()
    #     vec_norm.ob_rms = ob_rms

    eval_episode_rewards = []
    r_file = open( savedir +'/'+ 'clean'+ ".txt", "a")

    obs = eval_envs.reset()

    eval_recurrent_hidden_states = torch.zeros(
        num_processes, actor_critic.recurrent_hidden_state_size, device=device)
    eval_masks = torch.zeros(num_processes, 1, device=device)

    while len(eval_episode_rewards) < 30:
        obs_shape = obs.size()[1:]
        obs_shape = obs_shape[0]*obs_shape[1]*obs_shape[2] #4*84*84

        with torch.no_grad():

            _, action, _, eval_recurrent_hidden_states,_ = actor_critic.act(
                obs,
                eval_recurrent_hidden_states,
                eval_masks,
                deterministic=True)


        # Obser reward and next obs
        obs, rew, done, infos = eval_envs.step(action)

        #record
#         print("Evaluate time {}  1st process learner action {} AC action {} reward {}\n".format(j, action[0].data, action_true[0].data, rew[0].data), file=r_file, flush=True)
        eval_masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float32,
            device=device)
        for info in infos:
            if 'episode' in info.keys():
                eval_episode_rewards.append(info['episode']['r'])

    eval_envs.close()

    print("Evaluate time {} Evaluation using {} episodes: mean reward {:.5f} \n".format(j,
        len(eval_episode_rewards), np.mean(eval_episode_rewards)),file=r_file,flush=True)
