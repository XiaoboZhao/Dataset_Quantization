import os
import time
import torch
import argparse
import numpy as np

import dq.methods as methods
import dq.datasets as datasets
from dq.nets import LeNet5
from util.utils import str_to_bool

from torchvision import datasets, transforms


def main():
    parser = argparse.ArgumentParser(description='Parameter Processing')

    # Basic arguments
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--model', type=str, default='ResNet18', help='model')
    parser.add_argument('--selection', type=str, default="Submodular", help="selection method")
    parser.add_argument('--num_exp', type=int, default=5, help='the number of experiments')
    parser.add_argument('--batch', type=int, default=128, help='the number of batch size for selection')
    parser.add_argument('--gpu', default=None, nargs="+", type=int, help='gpu id to use')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--print_freq', '-p', default=50, type=int, help='print frequency (default: 20)')
    parser.add_argument('--fraction', default=0.1, type=float, help='fraction of data to be selected (default: 0.1)')
    parser.add_argument('--seed', default=int(time.time() * 1000) % 100000, type=int, help="random seed")
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

    # Selecting
    parser.add_argument("--selection_epochs", "-se", default=40, type=int,
                        help="number of epochs whiling performing selection on full dataset")
    parser.add_argument('--selection_momentum', '-sm', default=0.9, type=float, metavar='M',
                        help='momentum whiling performing selection (default: 0.9)')
    parser.add_argument('--selection_weight_decay', '-swd', default=5e-4, type=float,
                        metavar='W', help='weight decay whiling performing selection (default: 5e-4)',
                        dest='selection_weight_decay')
    parser.add_argument('--selection_optimizer', "-so", default="SGD",
                        help='optimizer to use whiling performing selection, e.g. SGD, Adam')
    parser.add_argument("--selection_nesterov", "-sn", default=True, type=str_to_bool,
                        help="if set nesterov whiling performing selection")
    parser.add_argument('--selection_lr', '-slr', type=float, default=0.1, help='learning rate for selection')
    parser.add_argument("--selection_test_interval", '-sti', default=1, type=int, help=
        "the number of training epochs to be preformed between two test epochs during selection (default: 1)")
    parser.add_argument("--selection_test_fraction", '-stf', type=float, default=1.,
             help="proportion of test dataset used for evaluating the model while preforming selection (default: 1.)")
    parser.add_argument('--balance', default=True, type=str_to_bool,
                        help="whether balance selection is performed per class")
    parser.add_argument('--replace', action='store_true', default=False, help='whether the samples can be selected repeatedly')
    parser.add_argument('--pretrained', action='store_true', default=False, help='whether the select model is pretrained')

    # Algorithm
    parser.add_argument('--submodular', default="GraphCut", help="specifiy submodular function to use")
    parser.add_argument('--submodular_greedy', default="NaiveGreedy", help="specifiy greedy algorithm for submodular optimization")

    # Checkpoint and resumption
    parser.add_argument('--save_path', "-sp", type=str, default='', help='path to save results (default: do not save)')

    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    args.selection_batch = args.batch
    print(args)
    if args.save_path != "" and not os.path.exists(args.save_path):
        os.makedirs(args.save_path, exist_ok=True)
    if not os.path.exists(args.data_path):
        os.makedirs(args.data_path, exist_ok=True)

    if args.pretrained and args.dataset == 'MNIST':
        print('Training LeNet5 on full MNIST for 10 epochs as pretrained model')
        start_time = time.time()
        # Use the imported LeNet5 model
        model = LeNet5()

        # Define transformations
        transform = transforms.Compose([
            transforms.Resize((32, 32)),  # LeNet-5 expects 32x32 images
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        # Download and load MNIST dataset
        train_dataset = datasets.MNIST(root='./data_MNIST', train=True, download=True, transform=transform)

        # Create data loader
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch, shuffle=True, 
            num_workers=args.workers, pin_memory=True
        )

        # Setup training
        device = torch.device(args.device)
        model = model.to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

        model.train()

        # Train for 10 epochs
        for epoch in range(10):
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            print(f'Epoch {epoch+1}/10 completed')

        # Save the pretrained model
        pretrained_path = os.path.join('/kaggle/working', 'pretrained_lenet5_mnist.pth')
        torch.save(model.state_dict(), pretrained_path)
        end_time = time.time()
        print(f"Model pre-trained on full MNIST for 10 epochs in {end_time - start_time:.2f} seconds.")
        print(f'Pretrained model saved to {pretrained_path}')

    # conduct non-overlapping coreset selection for multiple times
    for exp in range(args.num_exp):
        print('\n================== Exp %d ==================\n' % exp)
        print(args)
        
        # Add these lines before the problematic line to see what's happening
        print(f"args.dataset: {args.dataset}")
        print(f"Available datasets: {list(datasets.__dict__.keys())}")
        print(f"Is function available: {args.dataset in datasets.__dict__}")

        if args.dataset in datasets.__dict__:
            dataset_func = datasets.__dict__[args.dataset]
            print(f"Type of function: {type(dataset_func)}")
            result = dataset_func(args.data_path)
            print(f"Number of returned values: {len(result) if isinstance(result, tuple) else 1}")

        channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test = datasets.__dict__[args.dataset] \
            (args.data_path)
        args.channel, args.im_size, args.num_classes, args.class_names = channel, im_size, num_classes, class_names

        torch.random.manual_seed(args.seed)

        # initialize the available indices
        if exp == 0:
            avail_indices = np.arange(len(dst_train))
            coreset_size = round(len(dst_train) * args.fraction)

        selection_args = dict(epochs=args.selection_epochs,
                              balance=args.balance,
                              greedy=args.submodular_greedy,
                              function=args.submodular,
                              torchvision_pretrain=args.pretrained
                              )

        # re-initialize the training set with the remaining indices
        dst_train = torch.utils.data.Subset(dst_train, avail_indices)
        fraction = coreset_size / len(avail_indices)
        print('Exp: {}, Dst Size: {}, Fraction: {}'.format(exp, len(dst_train), fraction))
        
        # selection fraction samples from the remaining indices
        method = methods.__dict__[args.selection](dst_train, args, fraction, args.seed, **selection_args)
        subset = method.select()
        mapped_indices = avail_indices[subset["indices"]]
        print('Exp: {}, Available indices: {}, Select Subset: {}'.format(exp, len(avail_indices), len(mapped_indices)))

        # save the selected indices
        select_save_path = os.path.join(
            args.save_path, 'select_indices_{}_exp{}.npy'.format(args.dataset, exp)
        )
        np.save(select_save_path, mapped_indices)
        if not args.replace:
            avail_indices = np.delete(avail_indices, subset["indices"])

        # directly save the last subset
        if not args.replace and exp == args.num_exp - 2 and abs(args.fraction * args.num_exp - 1) < 1e-5:
            select_save_path = os.path.join(
                args.save_path, 'select_indices_{}_exp{}.npy'.format(args.dataset, exp + 1)
            )
            np.save(select_save_path, avail_indices)
            return
        else:
            continue


if __name__ == '__main__':
    main()
