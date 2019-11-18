"""
Input Output tools for training and testing CNN model on CIFAR10 data.

@author: Payam Dibaeinia
"""
import torchvision.transforms as transforms
import torchvision
import torch

def build_data_loader(data_path, train_batch_size, test_batch_size):
    """
    TODO: Enable custom augmentations!
    """

    train_aug = transforms.Compose([
        transforms.RandomCrop(32, padding = 4),
        transforms.RandomHorizontalFlip(p=0.2),
        transforms.ToTensor(),
        # I followed the suggestions in the below link to normalize data
        # https://github.com/kuangliu/pytorch-cifar/issues/19
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    test_aug = transforms.Compose([
        transforms.ToTensor(),
        # I followed the suggestions in the below link to normalize data
        # https://github.com/kuangliu/pytorch-cifar/issues/19
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])


    trainset = torchvision.datasets.CIFAR10(root = data_path, train = True, download = True, transform = train_aug)
    testset = torchvision.datasets.CIFAR10(root = data_path, train = False, download = True, transform = test_aug)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size = train_batch_size, shuffle = True, num_workers = 2)
    testloader = torch.utils.data.DataLoader(testset, batch_size = test_batch_size, shuffle = False, num_workers = 2)

    return trainloader, testloader
