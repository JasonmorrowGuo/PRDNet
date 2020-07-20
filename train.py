from torch.utils.data import DataLoader
from torchvision import transforms
from progressBar import printProgressBar

import medicalDataLoader

from utils import *
from visdom import Visdom

import dill
import argparse
from prdnet import resnet101,resnet50,resnet152,resnet34,resnet18
from optimizer import Adam



def weights_init(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.xavier_normal(m.weight.data)
    elif type(m) == nn.BatchNorm2d:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

#multi gpus
device_ids = [0]
def runTraining(args):
    print('-' * 40)
    print('~~~~~~~~  Starting the training... ~~~~~~')
    print('-' * 40)

    batch_size = args.batch_size
    batch_size_val = 1
    lr = args.lr

    epoch = args.epochs
    root_dir = '/media/hotchieh/GHDISK/服务器备份/semantic/MR3'
    model_dir = 'model'
    print(' Dataset: {} '.format(root_dir))
    transform = transforms.Compose([
        transforms.CenterCrop((256, 256)),
        transforms.ToTensor()
    ])

    mask_transform = transforms.Compose([
        transforms.CenterCrop((256, 256)),
        transforms.ToTensor()
    ])

    train_set = medicalDataLoader.MedicalImageDataset('train',
                                                      root_dir,
                                                      transform=transform,
                                                      mask_transform=mask_transform,
                                                      augment=True,
                                                      equalize=False)

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              num_workers=5,
                              shuffle=True)

    val_set = medicalDataLoader.MedicalImageDataset('val',
                                                    root_dir,
                                                    transform=transform,
                                                    mask_transform=mask_transform,
                                                    equalize=False)

    val_loader = DataLoader(val_set,
                            batch_size=batch_size_val,
                            num_workers=5,
                            shuffle=False)

                                                                    
    # Initialize
    net = resnet101(pretrained=False)
    #net = torch.nn.DataParallel(net, device_ids=device_ids)
    print(" Model Name: {}".format(args.modelName))

    net.apply(weights_init)

    softMax = nn.Softmax()
    CE_loss = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        net.cuda()
        softMax.cuda()
        CE_loss.cuda()

    optimizer = Adam(net.parameters(), lr=lr, betas=(0.9, 0.99), amsgrad=False)

    BestDice, BestEpoch = 0, 0

    Losses = []
    dataset = args.dataset
    print("~~~~~~~~~~~ Starting the training ~~~~~~~~~~")
    for i in range(epoch):
        net.train()
        lossVal = []
        totalImages = len(train_loader)
        for j, data in enumerate(train_loader):
            image, labels, img_names = data
            print(data[0].shape)
            print(data[1].shape)
            #print(data[2])

            # prevent batchnorm error for batch of size 1
            if image.size(0) != batch_size:
                continue

            optimizer.zero_grad()
            MRI = to_var(image)
            Segmentation = to_var(labels)

            ################### Train ###################
            net.zero_grad()
            segmentation_prediction = net(MRI)
            # It needs the logits, not the softmax
            Segmentation_class = getTargetSegmentation(Segmentation)
            # Cross-entropy loss
            loss = CE_loss(segmentation_prediction, Segmentation_class)

            loss.backward()
            optimizer.step()
            loss_visual = loss.cpu().data.numpy()

            viz.line([loss_visual], [i], win="LOSS {}".format(dataset), update='append')

            lossVal.append(loss_visual)
            printProgressBar(j + 1, totalImages,
                             prefix="[Training] Epoch: {} ".format(i),
                             length=15,
                             suffix=" Loss: {:.4f} ".format(
                                 loss_visual))

      
        printProgressBar(totalImages, totalImages,
                             done="[Training] Epoch: {}, Loss: {:.4f}".format(i,np.mean(lossVal)))
       
        # Save statistics
        modelName = args.modelName

        Losses.append(np.mean(lossVal))

        #prevent validation error
        if i > 0:

            dice, ravd, assd, mssd = inference(net, val_loader, args.dataset)

            currentDice = dice
            #viz.line([currentDice], [i], win='currentDice-cbam', update='append')
            viz.line([dice], [i], win="DICE {}".format(dataset), update='append')
            viz.line([assd], [i], win="ASSD {}".format(dataset), update='append')
            viz.line([[ravd, mssd]], [i], win="RAVD and MSSD {}".format(dataset), update='append')
            print("[val] metrics: (1): {:.4f} (2): {:.4f}  (3): {:.4f} (4): {:.4f}".format(dice,ravd,assd,mssd)) # MRI

            if currentDice > BestDice:
                BestDice = currentDice

                BestEpoch = i

                if currentDice > 0.40:

                    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Saving best model..... ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                    if not os.path.exists(model_dir):
                        os.makedirs(model_dir)
                    torch.save(net.state_dict(), os.path.join(model_dir, "Best_" + modelName + ".pth"),pickle_module=dill)

            print("###                                                       ###")
            print("###    Best Dice: {:.4f} at epoch {} with Dice: {:.4f} Ravd: {:.4f} Assd(3): {:.4f} Mssd(4): {:.4f}   ###".format(BestDice, BestEpoch, dice,ravd,assd,mssd))
            # print("###    Best Dice in 3D: {:.4f} with Dice(1): {:.4f} Dice(2): {:.4f} Dice(3): {:.4f} Dice(4): {:.4f} ###".format(np.mean(BestDice3D),BestDice3D[0], BestDice3D[1], BestDice3D[2], BestDice3D[3] ))
            print("###                                                       ###")



        if i % (BestEpoch + 50) == 0:
            for param_group in optimizer.param_groups:
                lr = lr*0.5
                param_group['lr'] = lr
                print(' ----------  New learning Rate: {}'.format(lr))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    #os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    parser.add_argument("--dataset",default="MR3",type=str)
    parser.add_argument("--modelName",default="MR3",type=str)
    parser.add_argument('--batch_size',default=12,type=int)
    parser.add_argument('--epochs',default=250,type=int)
    parser.add_argument('--lr',default=0.001,type=float)
    args=parser.parse_args()
    # visualization
    viz = Visdom()
    dataset = args.dataset
    # viz.line([0.], [0], win='currentDice-cbam', opts=dict(title='currentDice-cbam'))
    viz.line([[0., 0.]], [0], win="RAVD and MSSD {}".format(dataset),
             opts=dict(title="RAVD and MSSD {}".format(dataset), legend=['RAVD', 'MSSD']))
    viz.line([0.], [0], win="LOSS {}".format(dataset), opts=dict(title="LOSS {}".format(dataset)))
    viz.line([0.], [0], win="DICE {}".format(dataset), opts=dict(title="DICE {}".format(dataset)))
    viz.line([0.], [0], win="ASSD {}".format(dataset), opts=dict(title="ASSD {}".format(dataset)))
    runTraining(args)
