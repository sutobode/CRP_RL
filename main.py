from datetime import datetime
import argparse
import torch
from trainer import train, save_log, initialize
from benchmarks.benchmarks import solve_benchmarks


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--htr', action='store_true', help='Enable HTR mode (learned target selection)')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--log_path', type=str, default=f"./results/{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument('--load_model_path', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch_num', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--pomo_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--baseline', type=str, default='proposed')
    parser.add_argument('--instance_type', type=str, default='random')
    parser.add_argument('--n_layouts_per_batch', type=int, default=4)
    parser.add_argument('--min_n_containers', type=int, default=35)
    parser.add_argument('--max_n_containers', type=int, default=70)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_encode_layers', type=int, default=3)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--ff_hidden', type=int, default=512)
    parser.add_argument('--tanh_c', type=float, default=10)
    parser.add_argument('--lstm', action='store_true', default=True)
    parser.add_argument('--bay_embedding', action='store_true', default=True)
    parser.add_argument('--online', action='store_true')
    parser.add_argument('--online_known_num', type=int, default=None)
    return parser.parse_args()


def main():
    args = get_args()
    model, optimizer, clock = initialize(args)
    clock = save_log(args, -1, None, model, clock)
    solve_benchmarks(model, -1, args, instance_types=['random'])

    for epoch in range(args.epochs):
        train_loss = train(model, optimizer, args, epoch)
        clock = save_log(args, epoch, train_loss, model, clock)
        if (epoch + 1) % 1 == 0:
            solve_benchmarks(model, epoch, args, instance_types=['random'])


if __name__ == "__main__":
    main()
