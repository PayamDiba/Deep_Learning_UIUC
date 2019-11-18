"""
Input Output tools for training and testing on CIFAR10 data.

@author: Payam Dibaeinia
"""
import torchvision.transforms as transforms
import torchvision
import torch

def build_data_loader(data_path, train_batch_size, test_batch_size):
    """
    TODO: Enable custom augmentations!
    """

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.7, 1.0), ratio=(1.0,1.0)),
        transforms.ColorJitter(
                brightness=0.1*torch.randn(1),
                contrast=0.1*torch.randn(1),
                saturation=0.1*torch.randn(1),
                hue=0.1*torch.randn(1)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    transform_test = transforms.Compose([
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


    trainset = torchvision.datasets.CIFAR10(root = data_path, train = True, download = True, transform = transform_train)
    testset = torchvision.datasets.CIFAR10(root = data_path, train = False, download = True, transform = transform_test)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size = train_batch_size, shuffle = True, num_workers = 8)
    testloader = torch.utils.data.DataLoader(testset, batch_size = test_batch_size, shuffle = False, num_workers = 8)

    return trainloader, testloader
