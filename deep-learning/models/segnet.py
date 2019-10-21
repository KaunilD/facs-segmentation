import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
class SegNet(nn.Module):
    def __init__(self, in_ch=1, num_classes=2, debug=False):
        super(SegNet, self).__init__()

        self.in_ch = in_ch
        self.out_ch = num_classes
        self.debug = debug
        # 1, 16, 32

        self.encoder_1 = nn.Sequential(
            nn.Conv2d(
                in_channels = self.in_ch, out_channels=16,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(16)
        )
        self.encoder_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16, out_channels=32,
                kernel_size=2,
                padding=1
            ),
            nn.BatchNorm2d(32)
        )
        self.encoder_3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32, out_channels=64,
                kernel_size=1,
                padding=1
            ),
            nn.BatchNorm2d(64)
        )

        # Decoder layers
        self.decoder_3 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=64, out_channels=32,
                kernel_size=1,
                padding=1
            ),
            nn.BatchNorm2d(32)
        )
        self.decoder_2 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=32, out_channels=16,
                kernel_size=2,
                padding=1
            ),
            nn.BatchNorm2d(16)
        )
        self.decoder_1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=16, out_channels=2,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(2)
        )

    def forward(self, x):
        # Encoder

        # Encoder Stage - 1
        size_1 = x.size()
        x_1 = F.relu(self.encoder_1(x))
        x_1, indices_1 = F.max_pool2d(x_1, kernel_size=2, stride=2, return_indices=True)

        # Encoder Stage - 2
        size_2 = x_1.size()
        x_2 = F.relu(self.encoder_2(x_1))
        x_2, indices_2 = F.max_pool2d(x_2, kernel_size=2, stride=2, return_indices=True)

        # Encoder Stage - 3
        size_3 = x_2.size()
        x_3 = F.relu(self.encoder_3(x_2))
        x_3, indices_3 = F.max_pool2d(x_3, kernel_size=2, stride=2, return_indices=True)

        if self.debug:
            print("size_1: {}".format(size_1))
            print("size_2: {}".format(size_2))
            print("size_3: {}".format(size_3))
            print()

        # Decoder

        size_d = x_3.size()
        # Decoder Stage - 5
        x_3d = F.max_unpool2d(x_3, indices_3, kernel_size=2, stride=2, output_size=size_3)
        x_3d = F.relu(self.decoder_3(x_3d))
        size_3d = x_3d.size()
        # Decoder Stage - 4
        x_2d = F.max_unpool2d(x_3d, indices_2, kernel_size=2, stride=2, output_size=size_2)
        x_2d = F.relu(self.decoder_2(x_2d))
        size_2d = x_2d.size()
        # Decoder Stage - 1
        x_1d = F.max_unpool2d(x_2d, indices_1, kernel_size=2, stride=2, output_size=size_1)
        x_1d = F.relu(self.decoder_1(x_1d))
        size_1d = x_1d.size()

        x_softmax = F.softmax(x_1d, size=1)

        if self.debug:
            print("size_d: {}".format(size_d))
            print()
            print("size_3d: {}".format(size_3d))
            print("size_2d: {}".format(size_2d))
            print("size_1d: {}".format(size_1d))


        return x_1d, x_softmax
