# Code for "AMC: AutoML for Model Compression and Acceleration on Mobile Devices"
# Yihui He*, Ji Lin*, Zhijian Liu, Hanrui Wang, Li-Jia Li, Song Han
# {jilin, songhan}@mit.edu

import os
import numpy as np
import argparse
from copy import deepcopy
import torch
torch.backends.cudnn.deterministic = True

from env.channel_pruning_env import ChannelPruningEnv
from lib.ddpg import Agent
# from lib.td3 import Agent
from lib.utils import get_output_folder

from tensorboardX import SummaryWriter
import time
# from models_.mobilenet import MobileNet

def parse_args():
    parser = argparse.ArgumentParser(description='AMC search script')

    parser.add_argument('--job', default='train', type=str, help='support option: train/export')
    parser.add_argument('--suffix', default=None, type=str, help='suffix to help you remember what experiment you ran')
   
    # ----------------------------------env-------------------------------------#
   
    parser.add_argument('--model', default='resnet56', type=str, help='model to prune')
    parser.add_argument('--dataset', default='cifar10', type=str, help='dataset to use (cifar/imagenet)')
    parser.add_argument('--data_root', default='H:/share/cifar10/', type=str, help='dataset path')
    parser.add_argument('--preserve_ratio', default=0.5, type=float, help='preserve ratio of the model')
    parser.add_argument('--lbound', default=0.2, type=float, help='minimum preserve ratio')
    parser.add_argument('--rbound', default=1., type=float, help='maximum preserve ratio')
    parser.add_argument('--reward', default='acc_reward', choices=['acc_reward', 'acc_flops_reward'], type=str, help='Setting the reward')
    parser.add_argument('--acc_metric', default='acc1', type=str, help='use acc1 or acc5')
    parser.add_argument('--use_real_val', dest='use_real_val', action='store_true')
    parser.add_argument('--ckpt_path', default='./checkpoints/resnet56_cifar10_baseline.pth', type=str, help='manual path of checkpoint')#'./ckpt/mobilenet_cifar10_1.pth.tar'
    # parser.add_argument('--pruning_method', default='cp', type=str,    #'./ckpt/mobilenet_cifar10_1.pth.tar'
    #                     help='method to prune (fg/cp for fine-grained and channel pruning)')
    # only for channel pruning
    parser.add_argument('--n_calibration_batches', default=60, type=int,
                        help='n_calibration_batches')
    parser.add_argument('--n_points_per_layer', default=10, type=int,
                        help='method to prune (fg/cp for fine-grained and channel pruning)')
    parser.add_argument('--channel_round', default=8, type=int, help='Round channel to multiple of channel_round')
    
    # -------------------------------- ddpg ----------------------------------------#
    
    parser.add_argument('--agent_name', default='DDPG', type=str, help='ddpg/td3')
    parser.add_argument('--hidden1', default=300, type=int, help='hidden num of first fully connect layer')
    parser.add_argument('--hidden2', default=300, type=int, help='hidden num of second fully connect layer')
    parser.add_argument('--lr_c', default=1e-3, type=float, help='learning rate for actor')
    parser.add_argument('--lr_a', default=1e-4, type=float, help='learning rate for actor')
    parser.add_argument('--warmup', default=100, type=int,
                        help='time without training but only filling the replay memory')
    parser.add_argument('--discount', default=1., type=float, help='')
    parser.add_argument('--bsize', default=64, type=int, help='minibatch size') #replay buffer batch size
    parser.add_argument('--rmsize', default=100, type=int, help='memory size for each layer')
    parser.add_argument('--window_length', default=1, type=int, help='')
    parser.add_argument('--tau', default=0.01, type=float, help='moving average for target network')
    
    # -------------------------------- noise (truncated normal distribution)----------#
   
    parser.add_argument('--init_delta', default=0.5, type=float,
                        help='initial variance of truncated normal distribution')
    parser.add_argument('--delta_decay', default=0.95, type=float,
                        help='delta decay during exploration')
    
    # -------------------------------- training ---------------------------------------#
    
    parser.add_argument('--max_episode_length', default=1e9, type=int, help='')
    parser.add_argument('--output', default='./logs', type=str, help='')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--init_w', default=0.003, type=float, help='')
    parser.add_argument('--train_episode', default=300, type=int, help='train iters each timestep')
    parser.add_argument('--epsilon', default=50000, type=int, help='linear decay of exploration policy')
    parser.add_argument('--seed', default=None, type=int, help='random seed to set')
    parser.add_argument('--n_gpu', default=1, type=int, help='number of gpu to use')
    parser.add_argument('--n_worker', default=2, type=int, help='number of data loader worker')
    parser.add_argument('--data_bsize', default=128, type=int, help='number of data batch size')#test data batch size
    parser.add_argument('--resume', default='default', type=str, help='Resuming model path for testing')
    
    # --------------------------------- export ----------------------------------------#
   
    parser.add_argument('--ratios', default=None, type=str, help='ratios for pruning')
    parser.add_argument('--channels', default=None, type=str, help='channels after pruning')
    parser.add_argument('--export_path', default='./ckpt/mobilenet_0.5flops_cifar10_1_DDPG.pth.tar', type=str, help='path for exporting models')
    parser.add_argument('--use_new_input', dest='use_new_input', action='store_true', help='use new input feature')

    return parser.parse_args()


def get_model_and_checkpoint(model, dataset, checkpoint_path, n_gpu=1):
    if model == 'mobilenet' and dataset == 'cifar10':
        from models_.mobilenet import MobileNet
        net = MobileNet(n_class=10)# ,profile='0.5flops'
    elif model == 'mobilenetv2' and dataset == 'imagenet':
        from models_.mobilenet_v2 import MobileNetV2
        net = MobileNetV2(n_class=1000)
    elif model == 'resnet56' and dataset == 'cifar10':
        from models_.resnet import ResNet56
        net = ResNet56(num_classes=10)
    elif model == 'resnet20' and dataset == 'cifar10':
        from models_.resnet import ResNet20
        net = ResNet20(num_classes=10)
    else:
        raise NotImplementedError
    # if checkpoint_path is not None: ## gai
    #     sd = torch.load(checkpoint_path)
    #     if 'state_dict' in sd:  # a checkpoint but not a state_dict
    #         sd = sd['state_dict']
    #         sd = {k.replace('module.', ''): v for k, v in sd.items()}
    #         net.load_state_dict(sd)
    net = net.cuda()
    if n_gpu > 1:
        net = torch.nn.DataParallel(net, range(n_gpu))

    return net, deepcopy(net.state_dict())


def train(num_episode, agent, env, output, start_time):
    agent.is_training = True
    step = episode = episode_steps = 0
    episode_reward = 0.
    observation = None
    T = []  # trajectory
    

    ep_st = time.time()
    while episode < num_episode:  # counting based on episode
                
        
        # reset if it is the start of episode
        if observation is None:
            
            observation = deepcopy(env.reset())
            agent.reset(observation)
            

        # agent pick action ...
        if episode <= args.warmup:
            action = agent.random_action()
            # action = sample_from_truncated_normal_distribution(lower=0., upper=1., mu=env.preserve_ratio, sigma=0.5)
        else:
            action = agent.select_action(observation, episode=episode)

        # env response with next_observation, reward, terminate_info
        observation2, reward, done, info = env.step(action)
        observation2 = deepcopy(observation2)

        T.append([reward, deepcopy(observation), deepcopy(observation2), action, done])

        # fix-length, never reach here
        # if max_episode_length and episode_steps >= max_episode_length - 1:
        #     done = True

        # [optional] save intermideate model
        if episode % int(num_episode / 3) == 0:
            agent.save_model(output)

        # update
        step += 1
        episode_steps += 1
        episode_reward += reward
        observation = deepcopy(observation2)

        if done:  # end of episode
            print('#{}: episode_reward:{:.4f} acc: {:.4f}, ratio: {:.4f}'.format(episode, episode_reward,
                                                                                 info['accuracy'],
                                                                                 info['compress_ratio']))
            text_writer.write(
                '#{}: episode_reward:{:.4f} acc: {:.4f}, ratio: {:.4f}\n'.format(episode, episode_reward,
                                                                                 info['accuracy'],
                                                                                 info['compress_ratio']))
            final_reward = T[-1][0]
            # print('final_reward: {}'.format(final_reward))
            # agent observe and update policy
            for r_t, s_t, s_t1, a_t, done in T:
                agent.observe(final_reward, s_t, s_t1, a_t, done)
                if episode > args.warmup:
                    agent.update_policy()
            
            print('=======> one episode search time: %.2f\n'%(time.time() - ep_st))
            ep_st = time.time()
            #agent.memory.append(
            #    observation,
            #    agent.select_action(observation, episode=episode),
            #    0., False
            #)

            # reset
            observation = None
            episode_steps = 0
            episode_reward = 0.
            episode += 1
            T = []
            
            print('==>policy: {}'.format(str(env.strategy)))
            
            tfwriter.add_scalar('reward/last', final_reward, episode)
            tfwriter.add_scalar('reward/best', env.best_reward, episode)
            tfwriter.add_scalar('info/accuracy', info['accuracy'], episode)
            tfwriter.add_scalar('info/compress_ratio', info['compress_ratio'], episode)
            tfwriter.add_text('info/best_policy', str(env.best_strategy), episode)
            # record the preserve rate for each layer
            for i, preserve_rate in enumerate(env.strategy):
                tfwriter.add_scalar('preserve_rate/{}'.format(i), preserve_rate, episode)

            text_writer.write('best reward: {}\n'.format(env.best_reward))
            text_writer.write('best policy: {}\n'.format(env.best_strategy))
            
    text_writer.write('total search time: {}\n'.format(time.time()-start_time))
    text_writer.close()


def export_model(env, args):
    assert args.ratios is not None or args.channels is not None, 'Please provide a valid ratio list or pruned channels'
    assert args.export_path is not None, 'Please provide a valid export path'
    env.set_export_path(args.export_path)

    print('=> Original model channels: {}'.format(env.org_channels))
    if args.ratios:
        ratios = args.ratios.split(',')
        ratios = [float(r) for r in ratios]
        assert  len(ratios) == len(env.org_channels)
        channels = [int(r * c) for r, c in zip(ratios, env.org_channels)]
    else:
        channels = args.channels.split(',')
        channels = [int(r) for r in channels]
        ratios = [c2 / c1 for c2, c1 in zip(channels, env.org_channels)]
    print('=> Pruning with ratios: {}'.format(ratios))
    print('=> Channels after pruning: {}'.format(channels))

    for r in ratios:
        env.step(r)

    return


if __name__ == "__main__":
    args = parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
    
    start_time = time.time()
    
    model, checkpoint = get_model_and_checkpoint(args.model, args.dataset, checkpoint_path=args.ckpt_path,
                                                 n_gpu=args.n_gpu)

    env = ChannelPruningEnv(model, checkpoint, args.dataset,
                            preserve_ratio=1. if args.job == 'export' else args.preserve_ratio,
                            n_data_worker=args.n_worker, batch_size=args.data_bsize,
                            args=args, export_model=args.job == 'export', use_new_input=args.use_new_input)

    if args.job == 'train':
        # build folder and logs
        base_folder_name = '{}_{}_r{}_search_{}'.format(args.model, args.dataset, args.preserve_ratio, args.agent_name)
        if args.suffix is not None:
            base_folder_name = base_folder_name + '_' + args.suffix
        args.output = get_output_folder(args.output, base_folder_name)
        print('=> Saving logs to {}'.format(args.output))
        tfwriter = SummaryWriter(logdir=args.output)
        text_writer = open(os.path.join(args.output, 'log.txt'), 'w')#
        print('=> Output path: {}...'.format(args.output))

        nb_states = env.layer_embedding.shape[1]
        nb_actions = 1  # just 1 action here

        args.rmsize = args.rmsize * len(env.prunable_idx)  # for each layer
        print('** Actual replay buffer size: {}'.format(args.rmsize))

        agent = Agent(nb_states, nb_actions, args)
        train(args.train_episode, agent, env, args.output, start_time)
        
        print('=======> total search time: ', time.time()- start_time)
    
        
        # pruning, and export a pruned model
        best_p = env.best_strategy
        ratios = ''
        for i in best_p:
            ratios += (str(i)+',')
        
        args.ratios = ratios
        print(args.ratios)
        args.job = 'export'
        
        model, checkpoint = get_model_and_checkpoint(args.model, args.dataset, checkpoint_path=args.ckpt_path,
                                                  n_gpu=args.n_gpu)

        env = ChannelPruningEnv(model, checkpoint, args.dataset,
                            preserve_ratio=1. if args.job == 'export' else args.preserve_ratio,
                            n_data_worker=args.n_worker, batch_size=args.data_bsize,
                            args=args, export_model=args.job == 'export', use_new_input=args.use_new_input)
        export_model(env, args)        
        
    elif args.job == 'export':
    
        args.ratios = '1.0, 0.75, 0.5, 0.5625, 0.5, 0.46875, 0.4375, 0.4375, 0.4375, 0.375, 0.390625, 0.40625, 0.359375, 0.1484375, 0.1015625'
        export_model(env, args)
    else:
        raise RuntimeError('Undefined job {}'.format(args.job))

    








