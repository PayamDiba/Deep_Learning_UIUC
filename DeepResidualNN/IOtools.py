"""
Input Output tools for training and testing ResNet model on CIFAR100 and TinyImageNet

@author: Payam Dibaeinia
"""
import torchvision.transforms as transforms
import torchvision
import torch
from torchvision import datasets
import os

def build_CIFAR100_DataLoader(data_path, train_batch_size, test_batch_size, num_workers):
    train_aug = transforms.Compose([
        transforms.RandomCrop(32, padding = 4),
        transforms.RandomHorizontalFlip(p=0.2),
        transforms.ToTensor(),
        # I followed the normalization values used in the below code:
        # https://github.com/meliketoy/wide-resnet.pytorch/blob/master/config.py
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    test_aug = transforms.Compose([
        transforms.ToTensor(),
        # I followed the normalization values used in the below code:
        # https://github.com/meliketoy/wide-resnet.pytorch/blob/master/config.py
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    trainset = torchvision.datasets.CIFAR100(root=data_path, train=True, download=False, transform=train_aug)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=True, num_workers=num_workers)

    testset = torchvision.datasets.CIFAR100(root=data_path, train=False,download=False, transform=test_aug)
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False, num_workers=num_workers)

    return trainloader, testloader

def build_TinyImageNet_DataLoader(train_path, val_path, train_batch_size, val_batch_size, num_workers):

    def create_val_folder(val_dir):
        """
        This method is responsible for separating validation
        images into separate sub folders
        """
        # path where validation data is present now
        path = os.path.join(val_dir, 'images')
        # file where image2class mapping is present
        filename = os.path.join(val_dir, 'val_annotations.txt')
        fp = open(filename, "r") # open file in read mode
        data = fp.readlines() # read line by line
        '''
        Create a dictionary with image names as key and
        corresponding classes as values
        '''
        val_img_dict = {}
        for line in data:
            words = line.split("\t")
            val_img_dict[words[0]] = words[1]
        fp.close()
        # Create folder if not present, and move image into proper folder
        for img, folder in val_img_dict.items():
            newpath = (os.path.join(path, folder))
            if not os.path.exists(newpath): # check if folder exists
                os.makedirs(newpath)
            # Check if image exists in default directory
            if os.path.exists(os.path.join(path, img)):
                os.rename(os.path.join(path, img), os.path.join(newpath, img))
        return


    train_aug = transforms.Compose([
        transforms.RandomCrop(64, padding = 4),
        transforms.ToTensor(),
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    val_aug = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_dataset = datasets.ImageFolder(train_path, transform = train_aug)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=num_workers)

    if 'val_' in os.listdir(val_path)[0]:
        create_val_folder(val_path)
    else:
        pass

    val_dataset = datasets.ImageFolder(val_path, transform = val_aug)
    valloader = torch.utils.data.DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=num_workers)

    return trainloader, valloader
