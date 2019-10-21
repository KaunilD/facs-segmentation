import torch
import cv2
from PIL import Image
import numpy as np
import glob
import math
import os
import sys
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
import torch.utils.data as torch_data
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from models import segnet
import matplotlib.pyplot as plt
Image.MAX_IMAGE_PIXELS = None
class FACSDataset(torch_data.Dataset):
    def __init__(self,
                 input_list, target_list, split, debug = False, transform=None):
        """
        Args:
            input_list, target_list (list(str)): Directory with all the image/label pairs.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.input_list = input_list
        self.target_list = target_list

        self.transform = transform
        self.debug = debug

        self.images, self.masks = self.read_dir()

    def read_dir(self):
        tiles = [[], []]

        for idx, [img, msk] in enumerate(zip(self.input_list, self.target_list)):
            print('Reading item # {}/{}'.format(idx+1, len(images)), end='\r')
            image = cv2.imread(img , 0)
            image = np.resize(image, (1, 48, 48))
            mask = cv2.imread(msk, 0)

            _, mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)

            tiles[0].append(image)
            tiles[1].append(mask)

            del image
            del mask

        print()
        return tiles

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = [self.images[idx], self.masks[idx]]
        return sample

def focal_loss(output, target, device, gamma=2, alpha=0.5):
    n, c, h, w = output.size()
    criterion = nn.CrossEntropyLoss()
    criterion.to(device)
    logpt = -criterion(output, target.long())
    pt = torch.exp(logpt)
    if alpha is not None:
        logpt *= alpha
    loss = -((1 - pt) ** gamma) * logpt

    loss /= n

    return loss

def train(model, optimizer, criterion, device, dataloader):
    model.train()
    train_loss = 0.0
    tbar = tqdm(dataloader)
    num_samples = len(dataloader)
    for i, sample in enumerate(tbar):
        image, target = sample[0].float(), sample[1].float()
        image, target = image.to(device), target.float().to(device)
        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output[0], target, device)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
    return train_loss

def validate(model, criterion, device, dataloader):
    model.eval()
    val_loss = 0.0
    tbar = tqdm(dataloader)
    num_samples = len(dataloader)
    with torch.no_grad():
        for i, sample in enumerate(tbar):
            image, target = sample[0].float(), sample[1].float()
            image, target = image.to(device), target.float().to(device)

            output = model(image)
            loss = criterion(output[0], target, device)
            val_loss += loss.item()
            tbar.set_description('Val loss: %.3f' % (train_loss / (i + 1)))
    return val_loss


def test(model, device, dataloader):
    model.eval()
    val_loss = 0.0
    tbar = tqdm(dataloader)
    num_samples = len(dataloader)
    outputs = []
    with torch.no_grad():
        for i, sample in enumerate(tbar):
            image = sample[0].float()
            image = image.to(device)
            outputs.append(model(image))
            tbar.set_description('{}%'.format(int((i/num_samples)*100)))

    return outputs


if __name__=="__main__":
    print("torch.cuda.is_available()   =", torch.cuda.is_available())
    print("torch.cuda.device_count()   =", torch.cuda.device_count())
    print("torch.cuda.device('cuda')   =", torch.cuda.device('cuda'))
    print("torch.cuda.current_device() =", torch.cuda.current_device())
    print()

    epochs = 30
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_data_dir = '../dataset-creation/data_gt/input'
    target_data_dir = '../dataset-creation/data_gt/target'

    images = glob.glob(input_data_dir+'/*.png')
    target = glob.glob(target_data_dir+'/*.png')

    assert len(images) == len(target), "Input and labels dont match!"

    indices = np.random.permutation(len(images))

    images = [images[i] for i in indices]
    target = [target[i] for i in indices]

    train_dataset = FACSDataset(images[:25000], target[:25000], split='train', debug=False)
    val_dataset = FACSDataset(images[25000:], target[25000:], split='val', debug=False)

    train_dataloader = torch_data.DataLoader(train_dataset, num_workers=0, batch_size=32)
    val_dataloader = torch_data.DataLoader(val_dataset, num_workers=0, batch_size=64)


    model = segnet.SegNet(num_classes=2, debug=False)
    """
    if torch.cuda.device_count() > 1:
      print("Using ", torch.cuda.device_count(), " GPUs!")
      model = nn.DataParallel(model)
    """
    model.to(device)

    optimizer = torch.optim.SGD(
        lr=0.001,
        momentum=0.9,
        weight_decay=5e-4,
        nesterov=False,
        params=model.parameters()
    )

    criterion = focal_loss

    for epoch in range(epochs):
        train_loss = train(model, optimizer, criterion, device, train_dataloader)
        val_loss = validate(model, criterion, device, val_dataloader)
        outputs = test(model, device, val_dataloader)

        for idx, i in enumerate(outputs):
            i = i[1].cpu().numpy()

            mask = i[0][1]>i[0][0]

            mask = np.reshape(mask, (48, 48))
            plt.imsave( "./results/images/{}.png".format(idx), mask)

        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }

        model_save_str = './results/models/{}-{}-{}.{}'.format(
            "segnet", "bn2d", epoch, "pth"
        )

        torch.save(
            state,
            model_save_str
        )
        print(epoch, train_loss, val_loss)
