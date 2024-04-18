import os.path

import numpy as np
from torch.utils.data import Dataset
import SimpleITK as sitk
import cv2


def correct_dims(*images):
    corr_images = []
    # print(images)
    for img in images:
        if len(img.shape) == 2:
            corr_images.append(np.expand_dims(img, axis=2))
        else:
            corr_images.append(img)
    if len(corr_images) == 1:
        return corr_images[0]
    else:
        return corr_images


def get_image_from_video(filename):
    cap = cv2.VideoCapture(filename)

    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))

    fc = 0
    ret = True
    while (fc < frameCount and ret):
        ret, buf[fc] = cap.read()
        fc += 1
    buf = buf[:, :, :, :].transpose(0, 3, 1, 2)
    return buf


def resize_image_itk(itkimage, newSize, resamplemethod=sitk.sitkNearestNeighbor):
    resampler = sitk.ResampleImageFilter()

    originSize = itkimage.GetSize()  # 获取原图size
    originSpacing = itkimage.GetSpacing()  # 获取原图spacing
    newSize = np.array(newSize, dtype='uint32')
    factor = originSize / newSize
    newSpacing = originSpacing * factor

    resampler.SetReferenceImage(itkimage)  # 指定需要重新采样的目标图像
    resampler.SetSize(newSize.tolist())
    resampler.SetOutputSpacing(newSpacing.tolist())
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resamplemethod)
    itkimgResampled = resampler.Execute(itkimage)  # 得到重新采样后的图像
    itkimgResampled.SetOrigin(itkimage.GetOrigin())
    itkimgResampled.SetSpacing(itkimage.GetSpacing())
    itkimgResampled.SetDirection(itkimage.GetDirection())
    return itkimgResampled


class DatasetClassification(Dataset):
    def __init__(self, dir, transform=None):
        self.transform = transform  # using transform in torch!
        self.pos_dir = os.path.join(dir, "pos")
        self.neg_dir = os.path.join(dir, "neg")
        images = []

        neg_samples = os.listdir(self.neg_dir)
        neg_frames = 0
        for neg_sample in neg_samples:
            frames = get_image_from_video(os.path.join(self.neg_dir, neg_sample))  # (frames,B,H,W)
            # print(frames.shape)
            for frame in frames:
                images.append(frame)
            neg_frames += frames.shape[0]
        neg_label = np.zeros((neg_frames, 1))

        pos_samples = os.listdir(self.pos_dir)
        pos_frames = 0
        for pos_sample in pos_samples:
            frames = get_image_from_video(os.path.join(self.pos_dir, pos_sample, f"{pos_sample}.avi"))  # (frames,B,H,W)
            # print(frames.shape)
            for frame in frames:
                images.append(frame)
            pos_frames += frames.shape[0]
        pos_label = np.ones((pos_frames, 1))

        self.images = np.array(images)
        self.labels = np.array(np.concatenate((neg_label, pos_label), axis=0)).squeeze(1)
        print(f"Image:{self.images.shape}\tLabel:{self.labels.shape}")

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        image = correct_dims(self.images[idx])
        sample = {}
        if self.transform:
            image = self.transform(image)

        sample['image'] = image
        sample['label'] = self.labels[idx]
        return sample


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from augmentation import Transform2D
    tf = Transform2D(p_flip=1,crop=None)
    dataset = DatasetClassification("../dataset_sample", tf)
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    for idx,sample in enumerate(dataloader):
        if sample['label'].sum():
            image = sample['image']
            label = sample['label']
            print(image.shape,label.shape)
            plt.imshow(image[0][0],cmap="gray")
            plt.show()
            break
