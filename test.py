from torch.utils.data import DataLoader
from torchvision import transforms
from progressBar import printProgressBar


import medicalDataLoader

from utils import *
# from visdom import Visdom

import argparse
from prdnet import resnet50,resnet101,resnet18,resnet34,resnet152
# from unet import Unet
from optimizer import Adam
# from visdom import Visdom

# viz = Visdom()

def weights_init(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.xavier_normal(m.weight.data)
    elif type(m) == nn.BatchNorm2d:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def CRF_val(args):
    root_dir = '/media/hotchieh/GHDISK/服务器备份/semantic/MR2'
    batch_size_test_save = 1
    transform = transforms.Compose([
        transforms.CenterCrop((256, 256)),
        transforms.ToTensor()
    ])

    mask_transform = transforms.Compose([
        transforms.CenterCrop((256, 256)),
        transforms.ToTensor()
    ])

    test_set = medicalDataLoader.MedicalImageDataset('test',
                                                    root_dir,
                                                    transform=transform,
                                                    mask_transform=mask_transform,
                                                    equalize=False)

    test_loader_save_images = DataLoader(test_set,
                                        batch_size=batch_size_test_save,
                                        num_workers=4,
                                        shuffle=False)
    net = resnet101()
    # net = DAF_stack()
    pthfile = r'./model/Best_MR2.pth'
    net.load_state_dict(torch.load(pthfile))
    total = len(test_loader_save_images)
    net.eval()
    if torch.cuda.is_available():
        net.cuda()
    img_names_ALL = []
    val_path = "./test/{}".format(args.dataset)
    ground_dir = "./Data_3D/TESTGROUND/{}".format(args.dataset)
    dicom_dir = "./Data_3D/DICOM_test/{}".format(args.dataset)
    if not os.path.exists(val_path):
        os.makedirs(val_path)
    softMax = nn.Softmax().cuda()
    for i, data in enumerate(test_loader_save_images):
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


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    parser.add_argument("--dataset",default="MR2",type=str)
    parser.add_argument('--batch_size',default=1,type=int)
    parser.add_argument('--use_crf', default=False,
                        action='store_true', help='use crf or not')
    args=parser.parse_args()
    CRF_val(args)

