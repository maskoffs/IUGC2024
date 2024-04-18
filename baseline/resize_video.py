import cv2
import numpy as np
from moviepy.editor import *
from tqdm import tqdm
import SimpleITK as sitk


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


def resize_video(video_path):
    name = video_path.split('/')[-1] + ".avi"
    videocapture = cv2.VideoCapture(f"{video_path}/{name}")
    ori_rate, framenumber = videocapture.get(5), videocapture.get(7)

    print('帧速:{}\t 帧数{}\n'.format(ori_rate, framenumber))

    os.makedirs(f"{video_path}/low",exist_ok=True)
    video = cv2.VideoWriter(f"{video_path}/low/{name}", cv2.VideoWriter_fourcc(*"MJPG"), 24, (256, 256), False)
    for n in tqdm(range(int(framenumber))):
        videocapture.set(cv2.CAP_PROP_POS_FRAMES, n)
        ret, frame = videocapture.read()
        if frame is None:
            print("Error: Failed to read image")
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = sitk.GetImageFromArray(frame)
        frame = sitk.GetArrayFromImage(resize_image_itk(frame, (256, 256)))
        video.write(frame)
    video.release()


if __name__ == "__main__":
    path = "./dataset/val/pos"
    files = os.listdir(path)
    for file in files:
        resize_video(f"{path}/{file}")
