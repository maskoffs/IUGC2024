import pickle
import numpy as np
from os.path import isfile
import torch.nn as nn
import torch.nn.functional as F
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
    def __init__(self, in_ch=3, out_ch=3):
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


    def forward(self, x, isLabeled):
        """
        forward function should be designed
        """
        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)  # out  #(B,1024,32,32)

        cls_logit = self.norm(e5.mean([-2, -1]))
        cls_logit = self.head(cls_logit)  # (B,2)
        cls_logit = torch.softmax(cls_logit,dim=-1)
        cls = cls_logit[0][1] >= 0.5

        seg_flag = True if cls else False  # it means your model's prediction about classification

        if isLabeled:        # if the frame is labeled, it means this frame is standard plane to segment or measure. So whether the model's prediction about classification, it should perform segmentation.
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

            return cls_logit, seg, seg_flag
        else:
            return cls_logit, None, seg_flag  # If the frame is unlabeled, the return value of segmentation result is set to None.

class modelInter:
    def __init__(self):
        '''
        This constructor is supposed to initialize data members.
        Use triple quotes for function documentation.
        '''
        self.mean = None
        self.std = None

    def load(self, path):
        file = open(path, "rb")
        self.model = pickle.load(file)  # using pickle can load architecture and weight at the same time.
        # also you can use torch.load()

        return self

    def predict(self, X, isLabeled):
        """
        X: numpy array of shape (3,512,512), the input is one image/frame.
            If your model need to input a video or some frames, you can let X be a video and inform us while submitting your code.
        isLabeled: it's a sign which indicates whether the frame is labeled.
        """
        # self.model.eval()
        image = torch.tensor(X, dtype=torch.float).unsqueeze(0)

        cls, seg, seg_flag = self.model(image,isLabeled)  # cls (1,2)  seg (1,3,512,512)
        if seg is not None:
            seg = seg.squeeze(0).argmax(dim=0).detach().numpy()  # (512,512,3)
            """
                postprocess should be execute on segmentation, so that the predicted label will only retain the complete pubic symphysis (PS) and fetal head (FH) regions. 
                Naturally, the absence of post-processing methods is not an issue, but it may affect the recognition of the PS and FH contours in certain cases.
            """
        return cls.detach().numpy(), seg, seg_flag

    def save(self, path="./"):
        '''
        Save a trained model.
        '''
        pass
