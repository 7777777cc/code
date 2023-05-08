from os import lseek
import torch
import torch.nn as nn
import sklearn.metrics as skm
from a2c_ppo_acktr.arguments import get_args
import copy
import cv2
from torchvision import transforms as T
from pytorch_mutual_information_master import MutualInformation
from torch.autograd import Variable
import os

args = get_args()
device = torch.device("cuda:" + str(0)) # if args.cuda else "cpu")
MI = MutualInformation.MutualInformation(num_bins=256, sigma=0.4, normalize=True).to(device)

# trigger definition
    # trigger = torch.rand(4,2,2)*255
    # trigger.requires_grad_()
    # trigger_optimizer = torch.optim.SGD([{'params': trigger.data, 'lr':0.1}], momentum=0.9, dampening=0.9, weight_decay=0)

# definition of obs_action_grad function used later
#  def obs_action_grad(self, rollouts):
#         obs_shape = rollouts.obs.size()[2:]
#         action_shape = rollouts.actions.size()[-1]
#         num_steps, num_processes, _ = rollouts.rewards.size()

#         rollouts.obs.requires_grad_()
#         rollouts.obs.retain_grad()
#         input_image = rollouts.obs[:-1].view(-1, *obs_shape)
#         input_image.requires_grad_()
#         input_image.retain_grad()

#         values, action_log_probs, dist_entropy, rnn, dist = self.actor_critic.evaluate_dist(
#             input_image,
#             rollouts.recurrent_hidden_states[0].view(
#                 -1, self.actor_critic.recurrent_hidden_state_size),
#             rollouts.masks[:-1].view(-1, 1),
#             rollouts.actions.view(-1, action_shape))

#         values = values.view(num_steps, num_processes, 1)

#         advantages = rollouts.returns[:-1] - values
#         value_loss = advantages.pow(2).mean()
#         action_log_probs = action_log_probs.view(num_steps, num_processes, 1)

#         action_loss = -(advantages.detach() * action_log_probs).mean()

#         loss =  action_loss 

#         obs_grad = torch.autograd.grad(loss, input_image, create_graph=True)[0]

#         grads = obs_grad.reshape(5, 16, 4, 84, 84)
        
#         return grads, input_image, rollouts.obs, loss
#  obs+trigger -->classifier-->distribution-->[1,4] 
#                               supervised label[1]
#   -->ce loss trigger generator      

def poison(rollouts, location_i, location_j, step, pro, attacker, trigger, optimizer, mask, savedir):
    file = open(savedir + '/'+'loss_record' ".txt", "a")
    target = 0
    # print('\ntrigger before layer:\n', trigger)
    t = trigger.data
    trigger.requires_grad_()
    # mask.requires_grad_()
    # before = copy.deepcopy(rollouts)
    after = rollouts
    lamda = 0.01
    # after.obs[step, pro, :, int(location_i): int(location_i + trigger_size), int(location_j): int(location_j + trigger_size)] = trigger
    after.actions[step, pro, :] = torch.zeros((1, 1, 1)).int().cuda()
    after.rewards[step, pro, :] = 1
    after.compute_returns
    obs = after.obs[step, pro, :,:,:]
    # print(obs.size())
    trojan_obs = (1 - mask) * obs + mask * trigger
    eval_masks = torch.tensor([1.0] * 16, dtype=torch.float32, device=device)
    eval_recurrent_hidden_states = torch.zeros(16, attacker.actor_critic.recurrent_hidden_state_size, device=device)
    trojan_obs = trojan_obs.unsqueeze(0)
    y_target = torch.full((rollouts.actions.size(2),), target, dtype=torch.long).to(device)
    # after.obs.retain_grad()
    # _, _, _, _, y, action_log_probs= (attacker.obs_action_grad(after))
    _, _, _, eval_recurrent_hidden_states, dist = attacker.actor_critic.act(
                trojan_obs,
                eval_recurrent_hidden_states,
                eval_masks,
                deterministic=True)
    yhat = dist.logits #[80,4] rollouts.actions[5-step,16-processes,1]
    # yhat = y[step*pro,:].unsqueeze(0) #[1,4]
    # print(yhat,yhat.size())
    ls = nn.CrossEntropyLoss()
    loss = ls(yhat, y_target) #+ lamda * torch.sum(torch.abs(mask))
    # print('\ncheck the loss here!', loss)
    # import pdb;pdb.set_trace()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("loss: {} trigger grad: {}  trigger.sum: {}\n".format(loss, trigger.grad.sum(), trigger.sum()), file=file, flush=True)

    return trigger

