import cv2
import SimpleITK
import numpy as np
from ellipse import drawline_AOD


def onehot_to_mask(mask,resolution=512):

    ret = np.zeros([3, resolution, resolution])

    tmp = mask.copy()
    tmp[tmp == 1] = 255
    tmp[tmp == 2] = 0
    ret[1] = tmp
    tmp = mask.copy()
    tmp[tmp == 2] = 255
    tmp[tmp == 1] = 0
    ret[2] = tmp
    b = ret[0]
    r = ret[1]
    g = ret[2]
    ret = cv2.merge([b, r, g])
    mask = ret.transpose([0, 1, 2])
    return mask


def cal_aop(image: SimpleITK.Image) -> float:
    ellipse = None
    ellipse2 = None
    data = SimpleITK.GetArrayFromImage(image)
    aop_pred = np.array(onehot_to_mask(data)).astype(np.uint8)
    contours, _ = cv2.findContours(cv2.medianBlur(aop_pred[:, :, 1], 1), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    contours2, _ = cv2.findContours(cv2.medianBlur(aop_pred[:, :, 2], 1), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_NONE)
    maxindex1 = 0
    maxindex2 = 0
    max1 = 0
    max2 = 0
    flag1 = 0
    flag2 = 0
    for j in range(len(contours)):
        if contours[j].shape[0] > max1:
            maxindex1 = j
            max1 = contours[j].shape[0]
        if j == len(contours) - 1:
            approxCurve = cv2.approxPolyDP(contours[maxindex1], 1, closed=True)
            if approxCurve.shape[0] > 5:
                ellipse = cv2.fitEllipse(approxCurve)
            flag1 = 1
    for k in range(len(contours2)):
        if contours2[k].shape[0] > max2:
            maxindex2 = k
            max2 = contours2[k].shape[0]
        if k == len(contours2) - 1:
            approxCurve2 = cv2.approxPolyDP(contours2[maxindex2], 1, closed=True)
            if approxCurve2.shape[0] > 5:
                ellipse2 = cv2.fitEllipse(approxCurve2)
            flag2 = 1
    if flag1 == 1 and flag2 == 1 and ellipse2 != None and ellipse != None:
        aop = drawline_AOD(ellipse2, ellipse)


def run(image_path:str) -> float:
    image = SimpleITK.ReadImage(str(image_path))
    aop = cal_aop(image)
    return aop

if __name__ == '__main__':
    image_path = "./xxx.mha" # or "./xxx.png"
    aop = run(image_path)
    print(aop)
