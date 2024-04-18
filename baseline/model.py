from __future__ import print_function, division
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch


class conv_block(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class U_Net(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, mode="seg"):
        super(U_Net, self).__init__()
        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.mode = mode

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

        self.norm = nn.LayerNorm(1024, eps=1e-6)  # final norm layer
        self.head = nn.Linear(1024, 2)
        # self.activation = nn.Sigmoid()

        if mode == "seg":
            for param in self.norm.parameters():
                param.requires_grad = False
            for param in self.head.parameters():
                param.requires_grad = False
        else:
            for param in self.Maxpool1.parameters():
                param.requires_grad = False
            for param in self.Maxpool2.parameters():
                param.requires_grad = False
            for param in self.Maxpool3.parameters():
                param.requires_grad = False
            for param in self.Maxpool4.parameters():
                param.requires_grad = False
            for param in self.Conv1.parameters():
                param.requires_grad = False
            for param in self.Conv2.parameters():
                param.requires_grad = False
            for param in self.Conv3.parameters():
                param.requires_grad = False
            for param in self.Conv4.parameters():
                param.requires_grad = False
            for param in self.Conv5.parameters():
                param.requires_grad = False
            for param in self.Up2.parameters():
                param.requires_grad = False
            for param in self.Up3.parameters():
                param.requires_grad = False
            for param in self.Up4.parameters():
                param.requires_grad = False
            for param in self.Up5.parameters():
                param.requires_grad = False
            for param in self.Up_conv5.parameters():
                param.requires_grad = False
            for param in self.Up_conv4.parameters():
                param.requires_grad = False
            for param in self.Up_conv3.parameters():
                param.requires_grad = False
            for param in self.Up_conv2.parameters():
                param.requires_grad = False
            for param in self.Conv.parameters():
                param.requires_grad = False

    def forward(self, x, mode="inference"):
        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)  # out  #(B,1024,32,32)
        if mode == "seg":
            d5 = self.Up5(e5)
            d5 = torch.cat((e4, d5), dim=1)

            d5 = self.Up_conv5(d5)

            d4 = self.Up4(d5)
            d4 = torch.cat((e3, d4), dim=1)
            d4 = self.Up_conv4(d4)

            d3 = self.Up3(d4)
            d3 = torch.cat((e2, d3), dim=1)
            d3 = self.Up_conv3(d3)

            d2 = self.Up2(d3)
            d2 = torch.cat((e1, d2), dim=1)
            d2 = self.Up_conv2(d2)

            out = self.Conv(d2)
        elif mode == "cls":
            out = self.norm(e5.mean([-2, -1]))
            out = self.head(out)
            # out = self.activation(out)
        elif mode == "inference":
            out = self.norm(e5.mean([-2, -1]))
            out = self.head(out)  # (B,2)
            cls = out.argmax(dim=-1)
            seg = None
            if cls:
                d5 = self.Up5(e5)
                d5 = torch.cat((e4, d5), dim=1)
                d5 = self.Up_conv5(d5)

                d4 = self.Up4(d5)
                d4 = torch.cat((e3, d4), dim=1)
                d4 = self.Up_conv4(d4)

                d3 = self.Up3(d4)
                d3 = torch.cat((e2, d3), dim=1)
                d3 = self.Up_conv3(d3)

                d2 = self.Up2(d3)
                d2 = torch.cat((e1, d2), dim=1)
                d2 = self.Up_conv2(d2)

                seg = self.Conv(d2)
            return out, seg
        return out


if __name__ == '__main__':

    # x = torch.randn((2, 3, 512, 512)).cpu()
    model = U_Net().cpu()
    model.load_state_dict(torch.load("./checkpoints/epoch65_val_acc_0.85322.pth",map_location="cpu"))
    # out = model(x, "cls")
    # print(out.shape)
    import pickle

    with open('model.pickle', 'wb') as f:
        pickle.dump(model, f)
