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
from models import segnet, uresnet
import matplotlib.pyplot as plt
Image.MAX_IMAGE_PIXELS = None
class FACSDataset(torch_data.Dataset):
    def __init__(self,
                 input_list, target_list,
                 split, debug = False,
                 transform=None):
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

            print('Reading item # {}/{}'.format(idx+1, len(self.input_list)), end='\r')
            image = cv2.imread(img)
            mask = cv2.imread(msk, 0)

            if os.path.getsize(img) == 0 or os.path.getsize(msk) == 0:
                continue

            image = cv2.resize(image, (224, 224))
            cv2.normalize(image, image, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

            cv2.normalize(mask, mask, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX)
            mask = cv2.resize(mask, (224, 224))

            """
            # to tensor. This should not be used.
            plt.imsave("image.png", image)
            plt.imsave("mask.png", mask)
            # pytorch CHANNEL first, if transforms contains
            image = np.moveaxis(image, 2, 0)
            mask = np.reshape(mask, (1, 224, 224))
            """

            tiles[0].append(image)
            tiles[1].append(mask)

            del image
            del mask

        print()
        return tiles
    def resize(self, image, width=None, height=None, inter=cv2.INTER_AREA):
        dim = None
        (h, w) = image.shape[:2]

        if width is None and height is None:
            return image

        if width is None:
            r = height / float(h)
            dim = (int(w * r), height)
        else:
            r = width / float(w)
            dim = (width, int(h * r))

        resized = cv2.resize(image, dim, interpolation=inter)

        return resized

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        sample = [self.transform(self.images[idx]), self.masks[idx]]

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
        image, target = image.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, target, device)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        tbar.set_description('Train loss:  %.3f' % (train_loss / (i + 1)))
    return train_loss

def validate(model, criterion, device, dataloader):
    model.eval()
    val_loss = 0.0
    tbar = tqdm(dataloader)
    num_samples = len(dataloader)
    with torch.no_grad():
        for i, sample in enumerate(tbar):
            image, target = sample[0].float(), sample[1].float()
            image, target = image.to(device), target.to(device)

            output = model(image)
            loss = criterion(output, target, device)
            val_loss += loss.item()
            tbar.set_description('Val loss:    %.3f' % (train_loss / (i + 1)))
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
            outputs.append([sample[0], model(image), sample[1]])
            tbar.set_description('Test progress:%.2f' % (train_loss / (i + 1)))

    return outputs


if __name__=="__main__":
    """
    """
    print("torch.cuda.is_available()   =", torch.cuda.is_available())
    print("torch.cuda.device_count()   =", torch.cuda.device_count())
    print("torch.cuda.device('cuda')   =", torch.cuda.device('cuda'))
    print("torch.cuda.current_device() =", torch.cuda.current_device())
    print()

    epochs = 400
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    input_data_dir = '../dataset-creation/data_gt_lfw/input'
    target_data_dir = '../dataset-creation/data_gt_lfw/target'

    images = glob.glob(input_data_dir+'/*.png')
    target = glob.glob(target_data_dir+'/*.png')

    #assert len(images) == len(target), "Input and labels dont match!"

    indices = np.random.permutation(10600)
    #indices = np.load("indices.npy")
    np.save("indices", indices)
    images = [images[i] for i in indices]
    target = [target[i] for i in indices]

    train_dataset = FACSDataset(
        images[:7000], target[:7000],
        split='train',
        debug=False,
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    )
    val_dataset = FACSDataset(
        images[7000:9000], target[7000:9000],
        split='val', debug=False,
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    )
    test_dataset = FACSDataset(
        images[9000:10600], target[9000:10600],
        split='test', debug=False,
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    )


    train_dataloader = torch_data.DataLoader(train_dataset, num_workers=0, batch_size=32)
    val_dataloader = torch_data.DataLoader(val_dataset, num_workers=0, batch_size=32)
    test_dataloader = torch_data.DataLoader(test_dataset, num_workers=0, batch_size=32)

    #checkpoint = torch.load("./models/segnet-bn2d-1.pth")

    model = uresnet.UResNet()
    #model.load_state_dict(checkpoint["model"])

    if torch.cuda.device_count() > 1:
      print("Using ", torch.cuda.device_count(), " GPUs!")
      model = nn.DataParallel(model)

    model.to(device)


    optimizer = torch.optim.SGD(
        lr=0.001,
        momentum=0.9,
        weight_decay=5e-4,
        nesterov=False,
        params=model.parameters()
    )
    #optimizer.load_state_dict(checkpoint["optimizer"])

    criterion = focal_loss

    for epoch in range(epochs):
        train_loss = train(model, optimizer, criterion, device, train_dataloader)
        val_loss = validate(model, criterion, device, val_dataloader)
        outputs = test(model, device, test_dataloader)

        counter = 0
        if epoch%2 == 0:
            print("Saving results")
            for bdx, b in enumerate(outputs):
                for idx , i in enumerate(zip(b[0], b[1], b[2])):


                    img = torch.clamp(i[0], 0, 1).cpu().numpy()
                    img = np.moveaxis(img, 0, -1)

                    pred = torch.clamp(i[1], 0, 1).cpu().numpy()
                    pred = np.moveaxis(pred, 0, -1)

                    gt = torch.clamp(i[2], 0, 1).cpu().numpy()


                    pred_3C = np.zeros((224, 224, 3), dtype=np.float32)
                    gt_3C = np.zeros((224, 224, 3), dtype=np.float32)

                    pred_3C[:, :, :2] = pred
                    gt_3C[:, :, 0] = gt

                    #Calcium@20
                    res = np.concatenate((img, pred_3C, gt_3C), axis=1)
                    plt.imsave( "./results/images/{}.png".format(counter), res)

                    counter+=1

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
