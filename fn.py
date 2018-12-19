import time
import numpy as np
import torch
import scipy.misc
from torchsample.transforms import SpecialCrop, Pad
import torch.nn.functional as F



def getTime(time1=0):
    if not time1:
        return time.time()
    else:
        interval = time.time() - time1
    return time.time(), interval

def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray


def im_to_torch(img):
    img = np.transpose(img, (2, 0, 1))  # C*H*W
    img = to_torch(img).float()
    if img.max() > 1:
        img /= 255
    return img


def cropBox(img, ul, br, resH, resW):
    ul = ul.int()
    br = br.int()
    lenH = max(br[1] - ul[1], (br[0] - ul[0]) * resH / resW)
    lenW = lenH * resW / resH
    if img.dim() == 2:
        img = img[np.newaxis, :]

    newDim = torch.IntTensor((img.size(0), int(lenH), int(lenW)))
    newImg = img[:, ul[1]:, ul[0]:]
    # Crop and Padding
    size = torch.IntTensor((br[1] - ul[1], br[0] - ul[0]))

    newImg = SpecialCrop(size, 1)(newImg)
    newImg = Pad(newDim)(newImg)
    # Resize to output

    v_Img = torch.unsqueeze(newImg, 0)
    # newImg = F.upsample_bilinear(v_Img, size=(int(resH), int(resW))).data[0]
    newImg = F.upsample(v_Img, size=(int(resH), int(resW)),
                        mode='bilinear', align_corners=True).data[0]
    return newImg
