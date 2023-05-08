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

args = get_args()
device = torch.device("cuda:" + str(0)) # if args.cuda else "cpu")
MI = MutualInformation.MutualInformation(num_bins=256, sigma=0.4, normalize=True).to(device)


# def poison(rollouts, location_i, location_j, step, pro, attacker, trigger, optimizer, trigger_layer):
def poison(rollouts, location_i, location_j, step, pro, attacker, trigger, optimizer,  j):
    print('\ntrigger before layer:\n', trigger)
    trigger.requires_grad_()
    trigger_size = args.trigger_size
    before = copy.deepcopy(rollouts)
    after = copy.deepcopy(rollouts)

    # sigmoid = nn.Sigmoid()
    # relu = nn.ReLU()
    # trigger = trigger.reshape(1, trigger_size * trigger_size)

    # trigger = trigger_layer(trigger) # trigger_layer: nn.Linear
    # trigger = trigger.reshape(trigger_size, trigger_size)

    # trigger = relu(trigger)
    # trigger = sigmoid(trigger) * 255

    after.obs[step, pro, :, int(location_i): int(location_i + trigger_size), int(location_j): int(location_j + trigger_size)] = trigger
    after.actions[step, pro, :] = torch.zeros((1, 1, 1)).int().cuda()
    after.rewards[step, pro, :] = 1
    after.compute_returns

    # print(trigger.requires_grad)
    after.obs.retain_grad()
    # before_grad, before_input, before_obs = (attacker.obs_action_grad(before))
    # after_grad, after_input, after_obs = (attacker.obs_action_grad(after))
    before_grad, _, _, _ = (attacker.obs_action_grad(before))
    after_grad, _, _, _ = (attacker.obs_action_grad(after))

#######tracing gradients
    after_grad.retain_grad()
    before_grad.retain_grad()

#######
    after_grads = after_grad[step, pro, 3].abs() * 255
    before_grads = before_grad[step, pro, 3].abs() * 255
    
    after_grads.retain_grad()
    before_grads.retain_grad()
    # torch.save(before_grad, 'before_grad.pt') 
    # torch.save(after_grad, 'after_grad.pt')

    ls = nn.MSELoss()
    loss = ls(after_grads, before_grads)

    # after_grad = after_grad.unsqueeze(0).unsqueeze(0)
    # before_grad = before_grad.unsqueeze(0).unsqueeze(0)


    # loss_MI = MI(before_grad, after_grad)[0]
    # loss_MI =1-loss_MI
    # print('losss:\n',loss,loss_MI)

    # img1 = cv2.imread('/home/CuiJing/Workspace/rl/1.jpg')
    # img2 = cv2.imread('/home/CuiJing/Workspace/rl/2.jpg')
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # after_grad.retain_grad()
    # before.obs.retain_grad()
    # if j %5 ==0:
    #     loss_MI.backward()
    #     optimizer.step()

    # check
    print('\ntrigger.grad:\n', trigger.grad)
    print('\nafter_grad.grads:\n',after_grad.grad.sum())
    print('\nafter.obs.grad:\n',after.obs.grad.sum())
    # print('\ntrigger:\n', trigger)
    print('\nloss:\n', loss)
    print('\nbefore grad',before_grad.sum())
    # import pdb;pdb.set_trace()
    # print('trigger:',trigger.data)
    import pdb; pdb.set_trace()
    return trigger.data

    '''
    # optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # trig = sigmoid(trigger).int()*255
    # import pdb;pdb.set_trace()
    rollouts.obs[step,pro,:,int(location_i):int(location_i+trigger_size),int(location_j):int(location_j+trigger_size)] = trigger
    return rollouts.obs[step,pro,:,:,:]
    trigger_size = args.trigger_size
    before = copy.deepcopy(rollouts)
    after = copy.deepcopy(rollouts)
    
    sigmoid= nn.Sigmoid()
    relu = nn.ReLU()
    trigger = trigger_layer(trigger.reshape(1,900)).reshape(30,30)
    trigger = relu(trigger)
    trigger = sigmoid(trigger)*255
    # print('trigger has grad',trigger.grad)
    # print('trigger grads:',trigger.requires_grad)
    # print('before obs grads:',after.obs[step,pro,3].grad)

    # after.obs[step,pro,:,int(location_i):int(location_i+trigger_size),int(location_j):int(location_j+trigger_size)] = after.obs[step,pro,:,int(location_i):int(location_i+trigger_size),int(location_j):int(location_j+trigger_size)].detach() 
    after.obs[step,pro,:,int(location_i):int(location_i+trigger_size),int(location_j):int(location_j+trigger_size)] += trigger
    # print('after obs grads:',before.obs[step,pro,3].grad)
    after.actions=torch.zeros((5,16,1)).int().cuda()
    print('start before',before.obs.sum())
    before_grad = sigmoid(attacker.obs_action_grad(before)[step,pro,3])*255
    print('finish before')
    # print('start after {} trigger {}'.format(after.obs.sum(),trigger.sum()))
    with torch.no_grad():
        _, source_action, _, _, dist = attacker.actor_critic_clean.act(
        after.obs[step], after.recurrent_hidden_states[step],
        after.masks[step])
    after_grad =sigmoid(attacker.obs_action_grad(after)[step,pro,3])*255
    # print('finish after')
    print('changed actions',after.actions[step,pro,:],before.actions[step,pro,:])
    
    # img1 = cv2.imread('/home/CuiJing/Workspace/rl/1.jpg')
    # img2 = cv2.imread('/home/CuiJing/Workspace/rl/2.jpg')
    # print('before grad',before_grad[int(location_i):int(location_i+trigger_size),int(location_j):int(location_j+trigger_size)].sum(),'after_grad',after_grad[int(location_i):int(location_i+trigger_size),int(location_j):int(location_j+trigger_size)].sum())
    loss = MI(before_grad, after_grad)[0]
    loss =1-loss
    # print('loss has grad', loss.requires_grad)
    # import pdb;pdb.set_trace()
    # print('loss',loss)

    print('loss',loss)
    '''