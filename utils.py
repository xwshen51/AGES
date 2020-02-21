import numpy as np
import os
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader


def draw_recon(x, x_recon):
    x_l, x_recon_l = x.tolist(), x_recon.tolist()
    result = [None] * (len(x_l) + len(x_recon_l))
    result[::2] = x_l
    result[1::2] = x_recon_l
    return torch.FloatTensor(result)


def make_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)


def write_config_to_file(config, save_path):
    with open(os.path.join(save_path, 'config.txt'), 'w') as file:
        for arg in vars(config):
            file.write(str(arg) + ': ' + str(getattr(config, arg)) + '\n')


def make_dataloader(args):

    test_loader = None
    if args.dataset == 'celeba':
        trans_f = transforms.Compose([
            transforms.CenterCrop(args.image_size*2),
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_set = datasets.CelebA(args.data_dir, split='train', download=True, transform=trans_f,)

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=False,
                                                   drop_last=False, num_workers=3)

    elif args.dataset == 'cifar':
        trans_f = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        train_set = datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=trans_f)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=3)

    elif args.dataset == 'imagenet':
        trans_f = transforms.Compose([
            transforms.Resize((73, 73)),
            transforms.CenterCrop(64),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        train_set = datasets.ImageFolder(args.data_dir, transform=trans_f)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=False,
                                                   num_workers=8)

    elif args.dataset == 'mnist':
        trans_f = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        train_set = datasets.MNIST(args.data_dir, train=True, download=False, transform=trans_f)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=False,
                                                   num_workers=3)

    elif args.dataset == 'mnist_stack':
        train_set = np.load(args.data_dir)
        train_set = (train_set - 0.5) / 0.5
        train_set = TensorDataset(torch.from_numpy(train_set))
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=False, num_workers=3)
    
    else:
        assert args.dataset == 'mog'
        base = args.mog_base
        num_class = base ** 2
        num_each_class = [10000] * num_class
        if args.mog_imbalance:
            num_each_class[1::2] = [500] * (num_class // 2)
        means = []
        for i in range(base):
            for j in range(base):
                means.append([(i - (base - 1) / 2) * 2, (j - (base - 1) / 2) * 2])
        std = args.mog_std
        x = torch.randn(sum(num_each_class), 2) * std
        y = torch.ones(sum(num_each_class))
        for i in range(num_class):
            x[sum(num_each_class[:i]):sum(num_each_class[:(i + 1)]), :] += torch.Tensor(means[i])
            y[sum(num_each_class[:i]):sum(num_each_class[:(i + 1)])] *= i
        train_set = TensorDataset(x, y)
        train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(dataset=train_set, batch_size=10000, shuffle=True)

    return train_loader, test_loader
