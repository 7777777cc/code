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
device = 'cuda'
MI = MutualInformation.MutualInformation(num_bins=256, sigma=0.4, normalize=True).to('cuda')


def poison(rollouts,location_i,location_j,step,pro,attacker,trigger,optimizer,trigger_layer):

    trigger_size = args.trigger_size
    before = copy.deepcopy(rollouts)
    after = copy.deepcopy(rollouts)
    sigmoid= nn.Sigmoid()
    relu = nn.ReLU()
    # trigger.requires_grad_()
    trigger = trigger.reshape(1,trigger_size*trigger_size)


    print('before layer',trigger)
    trigger_after = trigger_layer(trigger)
    trigger_after = trigger_after.reshape(trigger_size,trigger_size)

    trigger_after = relu(trigger_after)
    trigger_after = sigmoid(trigger_after)*255
    print('afer calculation',trigger_after)
    print('trigger has grad',trigger_after.requires_grad)
    # print('trigger grads:',trigger.requires_grad)
    # print('before obs grads:',after.obs[step,pro,3].grad)
    # import pdb;pdb.set_trace()
    # after.obs[step,pro,:,int(location_i):int(location_i+trigger_size),int(location_j):int(location_j+trigger_size)] = after.obs[step,pro,:,int(location_i):int(location_i+trigger_size),int(location_j):int(location_j+trigger_size)].detach() 

    # after.obs[step,pro,:,int(location_i):int(location_i+trigger_size),int(location_j):int(location_j+trigger_size)] = trigger_after
    after.obs[step,pro,:,:] = trigger_after
    after.obs.requires_grad_()
    before.obs.requires_grad_()
    # print(tmp.size())
    # after.obs.requires_grad_()

    # y_grad = autograd.grad(z, y, retain_graph=True)[0]
    # print('after obs grads:',before.obs[step,pro,3].grad)
    after.actions=torch.zeros((5,16,1)).int().cuda()
    # print('start before',before.obs.sum())
    # before.obs.requires_grad_()
    before_grad, before_input, before_obs = (attacker.obs_action_grad(before))
    before_grad = before_grad[step,pro,3].abs()*255
 
    # print('finish before')
    # # print('start after {} trigger {}'.format(after.obs.sum(),trigger.sum()))
    # with torch.no_grad():
    #     _, source_action, _, _, dist = attacker.actor_critic_clean.act(
    #     after.obs[step], after.recurrent_hidden_states[step],
    #     after.masks[step])
    # after.obs.requires_grad_()

    after_grad, after_input, after_obs =(attacker.obs_action_grad(after))
    after_grad =after_grad[step,pro,3].abs()*255

    print('after grad ?',after_grad.requires_grad)
    print ('parameters',trigger_layer.state_dict())
    # saliency= attacker.obs_action_grad(before)[step,pro,3].abs()*255
    # cv2.imwrite('before.jpg',before.obs[step,pro,3].detach().cpu().numpy())
    # cv2.imwrite('before_grad_L.jpg',saliency.detach().cpu().numpy())


    # torch.save(saliency,'before_grad_L.pt')
    # print('grad comparision',(before_grad-after_grad).sum())
    # import pdb;pdb.set_trace()
    # test1 = torch.zeros_like(before_grad)
    # test2 = torch.ones_like(before_grad)
    # # print('finish after')
    # # print('changed actions',after.actions[step,pro,:],before.actions[step,pro,:])
    
    # img1 = cv2.imread('/home/CuiJing/Workspace/rl/3.jpg',cv2.IMREAD_GRAYSCALE)
    # img2 = cv2.imread('/home/CuiJing/Workspace/rl/4.jpg',cv2.IMREAD_GRAYSCALE)
    # img1 = cv2.resize(img1,(300,300))
    # img2 = cv2.resize(img2,(300,300))
    # img1 = torch.tensor(img1).cuda()
    # img2 = torch.tensor(img2).cuda()

    # img1 = torch.transpose(img1,0,2)
    # img2 = torch.transpose(img2,0,2)
    # img1 = transform(img1)
    # img2 = transform(img2)
    # print(img2.shape)
    # import pdb;pdb.set_trace()
    # print('before grad',before_grad[int(location_i):int(location_i+trigger_size),int(location_j):int(location_j+trigger_size)].sum(),'after_grad',after_grad[int(location_i):int(location_i+trigger_size),int(location_j):int(location_j+trigger_size)].sum())
   
    loss = MI(after_grad,before_grad)[0]

    # import pdb;pdb.set_trace()
    print('loss',loss)
    optimizer.zero_grad()
    after_grad.retain_grad()
    trigger_after.retain_grad()
    before.obs[step,pro,3].retain_grad()
    before_input.retain_grad()
    before_obs.retain_grad()
    loss.backward()
    optimizer.step()

    print('after_grad grad:',after_grad.grad.sum())
    print('trigger grad:', trigger_after.grad)
    print('input image grad:', before_input.grad.sum())
    print('before obsevation grad', before_obs.grad.sum())
    print('before obs grad: ', before.obs[step,pro,3].grad)
    # trig = sigmoid(trigger).int()*255
    import pdb;pdb.set_trace()
    with torch.no_grad():
        tmp1 = before.obs[step,pro,:,int(location_i):int(location_i+trigger_size),int(location_j):int(location_j+trigger_size)] + trigger_after.int()
        before.obs[step,pro,:,int(location_i):int(location_i+trigger_size),int(location_j):int(location_j+trigger_size)] = tmp1
    return before.obs[step,pro,:,:,:], trigger_after.data









