import os
import numpy as np
import argparse
from copy import deepcopy
import torch
import json

torch.backends.cudnn.deterministic = True

from lib.utils import get_output_folder
from tensorboardX import SummaryWriter


def parse_args():
    parser = argparse.ArgumentParser(description='AMC DDPG with minimum noise floor')

    parser.add_argument('--suffix', default=None, type=str)

    # Model and data
    parser.add_argument('--train_csv', required=True, type=str)
    parser.add_argument('--val_csv', required=True, type=str)
    parser.add_argument('--model_path', required=True, type=str)
    parser.add_argument('--labels', nargs='+', required=True)

    # Compression settings
    parser.add_argument('--preserve_ratio', default=0.5, type=float)
    parser.add_argument('--lbound', default=0.05, type=float)
    parser.add_argument('--rbound', default=1., type=float)
    parser.add_argument('--reward', default='amc_corrected_reward', type=str)
    parser.add_argument('--acc_metric', default='micro_auroc', type=str)

    # Pruning parameters
    parser.add_argument('--n_calibration_batches', default=60, type=int)
    parser.add_argument('--n_points_per_layer', default=10, type=int)
    parser.add_argument('--channel_round', default=4, type=int)

    # Agent hyperparameters
    parser.add_argument('--hidden1', default=300, type=int)
    parser.add_argument('--hidden2', default=300, type=int)
    parser.add_argument('--lr_c', default=1e-3, type=float)
    parser.add_argument('--lr_a', default=1e-4, type=float)
    parser.add_argument('--warmup', default=30, type=int)
    parser.add_argument('--discount', default=1., type=float)
    parser.add_argument('--bsize', default=64, type=int)
    parser.add_argument('--rmsize', default=500, type=int)
    parser.add_argument('--window_length', default=1, type=int)
    parser.add_argument('--tau', default=0.01, type=float)

    # Exploration - KEY PARAMETERS
    parser.add_argument('--init_delta', default=0.5, type=float)
    parser.add_argument('--delta_decay', default=0.995, type=float)
    parser.add_argument('--delta_min', default=0.15, type=float, help='MINIMUM noise floor')
    parser.add_argument('--epsilon', default=50000, type=int)

    # Training
    parser.add_argument('--train_episode', default=1000, type=int)
    parser.add_argument('--output', default='./logs', type=str)
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--n_worker', default=16, type=int)
    parser.add_argument('--data_bsize', default=50, type=int)
    
    # Environment version
    parser.add_argument('--env_version', default='v4', type=str, choices=['v4', 'v5'])

    return parser.parse_args()


def get_model_and_checkpoint(model_path, labels):
    from src.modeling.models.seresnet18 import resnet18
    net = resnet18(in_channel=12, out_channel=len(labels))
    sd = torch.load(model_path)
    if 'state_dict' in sd:
        sd = sd['state_dict']
    elif 'model_state_dict' in sd:
        sd = sd['model_state_dict']
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    net.load_state_dict(sd)
    net = net.cuda()
    return net, deepcopy(net.state_dict())


def train(num_episode, agent, env, output, args, tfwriter, text_writer):
    agent.is_training = True
    best_reward = -float('inf')
    best_acc = 0
    best_ratio = 1.0
    best_strategy = None
    
    improvement_history = []
    
    print("\n" + "="*70)
    print("DDPG TRAINING WITH MINIMUM NOISE FLOOR")
    print("="*70)
    print(f"Environment: {args.env_version}")
    print(f"Episodes: {num_episode}")
    print(f"Warmup: {args.warmup}")
    print(f"Initial delta: {args.init_delta}")
    print(f"Delta decay: {args.delta_decay}")
    print(f"Delta MIN: {args.delta_min} <-- KEY: never goes below this!")
    print("="*70 + "\n")

    for episode in range(num_episode):
        observation = env.reset()
        agent.reset(observation)
        
        episode_reward = 0
        done = False
        
        while not done:
            if episode < args.warmup:
                action = agent.random_action()
            else:
                action = agent.select_action(observation, episode=episode)
            
            next_observation, reward, done, info = env.step(action)
            episode_reward += reward
            
            agent.observe(reward, observation, next_observation, action, done)
            
            if episode >= args.warmup and agent.memory.nb_entries >= args.bsize:
                agent.update_policy()
            
            observation = next_observation
        
        # Get final info
        final_acc = info.get('accuracy', 0)
        final_ratio = info.get('compress_ratio', 1.0)
        final_reward = reward
        
        # Track best
        is_new_best = False
        if final_reward > best_reward:
            best_reward = final_reward
            best_acc = final_acc
            best_ratio = final_ratio
            best_strategy = env.strategy.copy() if hasattr(env, 'strategy') else None
            is_new_best = True
            improvement_history.append((episode, final_reward, final_acc, final_ratio))
        
        # Calculate current delta (with minimum floor)
        if episode >= args.warmup:
            current_delta = args.init_delta * (args.delta_decay ** (episode - args.warmup))
            current_delta = max(current_delta, args.delta_min)  # Apply floor
        else:
            current_delta = args.init_delta
        
        # Print progress
        if is_new_best:
            print(f"Ep {episode:4d}: reward={final_reward:.4f}, acc={final_acc:.2f}%, "
                  f"ratio={final_ratio:.4f} *** NEW BEST ***")
        elif episode % 20 == 0:
            print(f"Ep {episode:4d}: reward={final_reward:.4f}, acc={final_acc:.2f}%, "
                  f"ratio={final_ratio:.4f} | best={best_reward:.4f} | δ={current_delta:.3f}")
        
        # Logging
        tfwriter.add_scalar('reward/final', final_reward, episode)
        tfwriter.add_scalar('reward/best', best_reward, episode)
        tfwriter.add_scalar('info/accuracy', final_acc, episode)
        tfwriter.add_scalar('info/compress_ratio', final_ratio, episode)
        tfwriter.add_scalar('search/delta', current_delta, episode)
        
        text_writer.write(f'{episode},{final_reward:.6f},{final_acc:.4f},{final_ratio:.6f}\n')
        text_writer.flush()
        
        if episode % 100 == 0 and episode > 0:
            agent.save_model(output)
    
    # Final summary
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"\nBest: reward={best_reward:.4f}, acc={best_acc:.2f}%, ratio={best_ratio:.4f}")
    print(f"\nImprovements ({len(improvement_history)}):")
    for ep, rew, acc, ratio in improvement_history[-15:]:
        print(f"  Episode {ep:4d}: reward={rew:.4f}, acc={acc:.2f}%, ratio={ratio:.4f}")
    
    # Save results
    results = {
        'best_reward': float(best_reward),
        'best_accuracy': float(best_acc),
        'best_ratio': float(best_ratio),
        'best_strategy': [float(x) for x in best_strategy] if best_strategy else None,
        'improvements': len(improvement_history),
        'delta_min': args.delta_min,
    }
    with open(os.path.join(output, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    agent.save_model(output)
    text_writer.close()
    
    return best_reward, best_strategy


if __name__ == "__main__":
    args = parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    # Load model
    model, checkpoint = get_model_and_checkpoint(args.model_path, args.labels)

    data_paths = {
        'train': args.train_csv,
        'val': args.val_csv
    }

    # Create environment
    if args.env_version == 'v5':
        from env.ecg_channel_pruning_env_v5 import ECGChannelPruningEnvV5
        env = ECGChannelPruningEnvV5(
            model, checkpoint, data_paths,
            preserve_ratio=args.preserve_ratio,
            n_data_worker=args.n_worker,
            batch_size=args.data_bsize,
            args=args,
            export_model=False,
            use_new_input=False
        )
    else:
        from env.ecg_channel_pruning_env_v4 import ECGChannelPruningEnv
        env = ECGChannelPruningEnv(
            model, checkpoint, data_paths,
            preserve_ratio=args.preserve_ratio,
            n_data_worker=args.n_worker,
            batch_size=args.data_bsize,
            args=args,
            export_model=False,
            use_new_input=False
        )

    # Setup output
    base_folder_name = f'ecg_ddpg_{args.env_version}_r{args.preserve_ratio}'
    if args.suffix:
        base_folder_name += '_' + args.suffix
    args.output = get_output_folder(args.output, base_folder_name)
    
    print(f'=> Output: {args.output}')
    tfwriter = SummaryWriter(logdir=args.output)
    text_writer = open(os.path.join(args.output, 'log.csv'), 'w')
    text_writer.write('episode,reward,accuracy,ratio\n')

    # Create agent with MINIMUM NOISE FLOOR
    from lib.agent_fixed import DDPG  # Use fixed agent!
    
    nb_states = env.layer_embedding.shape[1]
    nb_actions = 1
    args.rmsize = args.rmsize * env.n_prunable_layer
    
    print(f'=> Buffer size: {args.rmsize}')
    print(f'=> Layers: {env.n_prunable_layer}')
    print(f'=> Delta min: {args.delta_min}')

    agent = DDPG(nb_states, nb_actions, args)
    
    train(args.train_episode, agent, env, args.output, args, tfwriter, text_writer)
