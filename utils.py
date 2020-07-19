import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import os
import skimage.transform as skiTransf
from progressBar import printProgressBar
import scipy.io as sio
import pdb
import time
from os.path import isfile, join
from PIL import Image

from CHAOSmetrics import png_series_reader
from CHAOSmetrics import evaluate


#
# def getImageImageList(imagesFolder):
#     if os.path.exists(imagesFolder):
#        imageNames = [f for f in os.listdir(imagesFolder) if isfile(join(imagesFolder, f))]
#
#     imageNames.sort()
#
#     return imageNames

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


    
def getSingleImage(pred):
    # input is a 4-channels image corresponding to the predictions of the net
    # output is a gray level image (1 channel) of the segmentation with "discrete" values
    num_classes = 5
    Val = to_var(torch.zeros(num_classes))

    # Chaos MRI
    Val[1] = 0.24705882
    Val[2] = 0.49411765
    Val[3] = 0.7411765
    Val[4] = 0.9882353
    
    x = predToSegmentation(pred)
   
    out = x * Val.view(1, 5, 1, 1)

    return out.sum(dim=1, keepdim=True)


def predToSegmentation(pred):
    Max = pred.max(dim=1, keepdim=True)[0]
    x = pred / Max
    return (x == 1).float()




def getTargetSegmentation(batch):
    # input is 1-channel of values between 0 and 1
    # values are as follows : 0, 0.3137255, 0.627451 and 0.94117647
    # output is 1 channel of discrete values : 0, 1, 2 and 3
    
    denom = 0.24705882 # for Chaos MRI  Dataset this value

    return (batch / denom).round().long().squeeze()




def inference(net, img_batch):
    total = len(img_batch)
    net.eval()
    img_names_ALL = []
    val_path = './validation/MR1'
    ground_dir = './Data_3D/VALGROUND/MR1'
    dicom_dir = './Data_3D/DICOM_val/MR1'
    if not os.path.exists(val_path):
        os.makedirs(val_path)
    softMax = nn.Softmax().cuda()
    for i, data in enumerate(img_batch):
        printProgressBar(i, total, prefix="[Inference] Getting segmentations...", length=30)
        image, labels, img_names = data
        img_names_ALL.append(img_names[0].split('/')[-1].split('.')[0])

        MRI = to_var(image)

        segmentation_prediction = net(MRI)

        pred_y = softMax(segmentation_prediction)
        segmentation = getSingleImage(pred_y)

        str_1 = img_names[0].split('/Img/')
        str_subj = str_1[1]
        torchvision.utils.save_image(segmentation.data, os.path.join(val_path, str_subj))

    printProgressBar(total, total, done="[Inference] Segmentation Done !")

    # ======= Directories =======
    print("ground_dir：{}".format(ground_dir))
    # seg_dir = os.path.normpath(cwd + '/Data_3D/segmentation')
    seg_dir = val_path
    print("seg_dir：{}".format(seg_dir))
    print("dicom_dir：{}".format(dicom_dir))

    # ======= Volume Reading =======
    Vref = png_series_reader(ground_dir)
    Vseg = png_series_reader(seg_dir)
    print('Volumes imported.')
    # ======= Evaluation =======
    print('Calculating...')
    [dice, ravd, assd, mssd] = evaluate(Vref, Vseg, dicom_dir)
    print('DICE=%.3f RAVD=%.3f ASSD=%.3f MSSD=%.3f' % (dice, ravd, assd, mssd))

    return [dice, ravd, assd, mssd]

def inference_test(net, img_batch):
    total = len(img_batch)
    net.eval()
    img_names_ALL = []
    val_path = './test/MR2'
    ground_dir = './Data_3D/TESTGROUND/MR2'
    dicom_dir = './Data_3D/DICOM_test/MR2'
    if not os.path.exists(val_path):
        os.makedirs(val_path)
    softMax = nn.Softmax().cuda()
    for i, data in enumerate(img_batch):
        printProgressBar(i, total, prefix="[Inference] Getting segmentations...", length=30)
        image, labels, img_names = data
        img_names_ALL.append(img_names[0].split('/')[-1].split('.')[0])

        MRI = to_var(image)

        segmentation_prediction = net(MRI)

        pred_y = softMax(segmentation_prediction)
        segmentation = getSingleImage(pred_y)

        str_1 = img_names[0].split('/Img/')
        str_subj = str_1[1]
        torchvision.utils.save_image(segmentation.data, os.path.join(val_path, str_subj))

    printProgressBar(total, total, done="[Inference] Segmentation Done !")

    # ======= Directories =======
    print("ground_dir：{}".format(ground_dir))
    # seg_dir = os.path.normpath(cwd + '/Data_3D/segmentation')
    seg_dir = val_path
    print("seg_dir：{}".format(seg_dir))
    print("dicom_dir：{}".format(dicom_dir))

    # ======= Volume Reading =======
    Vref = png_series_reader(ground_dir)
    Vseg = png_series_reader(seg_dir)
    print('Volumes imported.')
    # ======= Evaluation =======
    print('Calculating...')
    [dice, ravd, assd, mssd] = evaluate(Vref, Vseg, dicom_dir)
    print('DICE=%.3f RAVD=%.3f ASSD=%.3f MSSD=%.3f' % (dice, ravd, assd, mssd))

    return [dice, ravd, assd, mssd]



class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).float()

