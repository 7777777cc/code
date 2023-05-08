import argparse

import torch


def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
        '--algo', default='a2c', help='algorithm to use: a2c | ppo | acktr')
    parser.add_argument(
        '--gail',
        action='store_true',
        default=False,
        help='do imitation learning with gail')
    parser.add_argument(
        '--gail-experts-dir',
        default='./gail_experts',
        help='directory that contains expert demonstrations for gail')
    parser.add_argument(
        '--gail-batch-size',
        type=int,
        default=128,
        help='gail batch size (default: 128)')
    parser.add_argument(
        '--gail-epoch', type=int, default=5, help='gail epochs (default: 5)')
    parser.add_argument(
        '--lr', type=float, default=7e-4, help='learning rate (default: 7e-4)')
    parser.add_argument(
        '--eps',
        type=float,
        default=1e-5,
        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.99,
        help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.99,
        help='discount factor for rewards (default: 0.99)')
    parser.add_argument(
        '--use-gae',
        action='store_true',
        default=False,
        help='use generalized advantage estimation')
    parser.add_argument(
        '--gae-lambda',
        type=float,
        default=0.95,
        help='gae lambda parameter (default: 0.95)')
    parser.add_argument(
        '--entropy-coef',
        type=float,
        default=0.01,
        help='entropy term coefficient (default: 0.01)')
    parser.add_argument(
        '--value-loss-coef',
        type=float,
        default=0.5,
        help='value loss coefficient (default: 0.5)')
    parser.add_argument(
        '--max-grad-norm',
        type=float,
        default=0.5,
        help='max norm of gradients (default: 0.5)')
    parser.add_argument(
        '--seed', type=int, default=0, help='random seed (default: 1)')
    parser.add_argument(
        '--cuda-deterministic',
        action='store_true',
        default=False,
        help="sets flags for determinism when using CUDA (potentially slow!)")
    parser.add_argument(
        '--num-processes',
        type=int,
        default=16,
        help='how many training CPU processes to use (default: 16)')
    parser.add_argument(
        '--num-steps',
        type=int,
        default=5,
        help='number of forward steps in A2C, i.e.the size of rollouts (default: 5)')
    parser.add_argument(
        '--ppo-epoch',
        type=int,
        default=4,
        help='number of ppo epochs (default: 4)')
    parser.add_argument(
        '--num-mini-batch',
        type=int,
        default=32,
        help='number of batches for ppo (default: 32)')
    parser.add_argument(
        '--clip-param',
        type=float,
        default=0.2,
        help='ppo clip parameter (default: 0.2)')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=100,
        help='log interval, one log per n updates (default: 10)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=1000,
        help='save interval, one save per n updates (default: 100)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=10000,
        help='eval interval, one eval per n updates (default: None)')
    parser.add_argument(
        '--num-env-steps',
        type=int,
        default=9*8e5,
        help='number of environment steps to train (default: 3e5)')
    parser.add_argument(
        '--env-name',
        default='BreakoutNoFrameskip-v4',
        help='environment to train on (default: BreakoutNoFrameskip-v4)')
    parser.add_argument(
        '--log-dir',
        default='/tmp/gym/',
        help='directory to save agent logs (default: /tmp/gym)')
    parser.add_argument(
        '--save-dir',
        default='./trained_models/',
        help='directory to save agent logs (default: ./trained_models/)')
    parser.add_argument(
        '--no-cuda',
        action='store_true',
        default=False,
        help='disables CUDA training')
    parser.add_argument(
        '--cuda-id',
        type=int,
        default=0,
        help='the id of gpu, if use cuda')
    parser.add_argument(
        '--use-proper-time-limits',
        action='store_true',
        default=False,
        help='compute returns taking into account time limits')
    parser.add_argument(
        '--recurrent-policy',
        action='store_true',
        default=False,
        help='use a recurrent policy')
    parser.add_argument(
        '--use-linear-lr-decay',
        action='store_true',
        default=False,
        help='use a linear schedule on the learning rate')


    # attack settings
    parser.add_argument('--stepsize', type=float, default=0.05)
    parser.add_argument('--maxiter', type=int, default=10)
    parser.add_argument('--radius', type=float, default=0.5)
    parser.add_argument('--radius-s', type=float, default=0.5)
    parser.add_argument('--radius-a', type=float, default=0.05)
    parser.add_argument('--radius-r', type=float, default=0.1)
    parser.add_argument('--frac', type=float, default=0.0)
    parser.add_argument('--type', type=str, default="wb", help="rand, wb, semirand")
    parser.add_argument('--aim', type=str, default="reward", help="reward, obs, action")
    parser.add_argument('--delta', type=float, default=0.1)

    parser.add_argument('--attack', dest='attack', action='store_true')
    parser.add_argument('--no-attack', dest='attack', action='store_false')
    parser.set_defaults(attack=False)

    parser.add_argument('--compute', dest='compute', action='store_true')
    parser.add_argument('--no-compute', dest='compute', action='store_false')
    parser.set_defaults(compute=False)
    parser.add_argument('--dist_thres', type=float, default=0.1)

    parser.add_argument('--max-episodes', type=int, default=1000)
    parser.add_argument('--run', type=int, default=-1)
    # file settings
    parser.add_argument('--logdir', type=str, default="logs/")
    parser.add_argument('--resdir', type=str, default="backupcheck/")
    parser.add_argument('--resdircj', type=str, default="backupcheck/")
    parser.add_argument('--resdirsam', type=str, default="backupcheck/")
    parser.add_argument('--moddir', type=str, default="models/")
    parser.add_argument('--loadfile', type=str, default="")

    # parser.add_argument(
    #     '--attack_condition',
    #     type=str,
    #     default ='belonging',
    #     help='way of defining the attack condition')
####backdoor
    parser.add_argument('--top-k', type=float, default=0.15)
    parser.add_argument('--trigger-size', type=int, default=3)
    parser.add_argument('--trigger-color', type=int, default=50)
    parser.add_argument('--reward-change', type=float, default=1)
    parser.add_argument('--limit', type=float, default=.01)
    parser.add_argument('--patch-size', type=int, default=20)
    parser.add_argument('--tuning-steps', type = int, default=200000)
    parser.add_argument('--position-steps', type=int, default=20000)

########

    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    assert args.algo in ['a2c', 'ppo', 'acktr']
    if args.recurrent_policy:
        assert args.algo in ['a2c', 'ppo'], \
            'Recurrent policy is not implemented for ACKTR'

    return args
