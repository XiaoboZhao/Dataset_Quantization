from torchvision import datasets, transforms
from torch import tensor, long


def CIFAR100(data_path):
    channel = 3
    im_size = (32, 32)
    num_classes = 100
    mean = [0.5071, 0.4867, 0.4408]
    std = [0.2675, 0.2565, 0.2761]

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    dst_train = datasets.CIFAR100(data_path, train=True, download=True, transform=transform)
    dst_test = datasets.CIFAR100(data_path, train=False, download=True, transform=transform)
    class_names = dst_train.classes
    dst_train.targets = tensor(dst_train.targets, dtype=long)
    dst_test.targets = tensor(dst_test.targets, dtype=long)
    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test
    