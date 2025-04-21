from torchvision import datasets, transforms
from torch import tensor, long


def MNIST(data_path):
    channel = 1
    im_size = (28, 28)
    num_classes = 10
    mean = [0.1307]
    std = [0.3081]

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(), 
        transforms.Normalize(mean=mean, std=std)])
    dst_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
    dst_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)
    class_names = dst_train.classes
    dst_train.targets = tensor(dst_train.targets, dtype=long)
    dst_test.targets = tensor(dst_test.targets, dtype=long)
    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test
