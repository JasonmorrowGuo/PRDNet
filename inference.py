

from utils import *
from visdom import Visdom
import dill
import argparse
# from unet import Unet
from prdnet import resnet50,resnet101,resnet18,resnet34,resnet152
# from unet import Unet
from optimizer import Adam
from PIL import Image
from visdom import Visdom
from data.utils import decode_seg_map_sequence
from torchvision.utils import save_image
from torchvision import transforms
# from thop import profile, clever_format
# from ptflops import get_model_complexity_info

vis = Visdom()


pic_path = './subj_2slice_12.png'
#pic = scipy.misc.imread(pic_path,mode='RGB')
pic = Image.open(pic_path)


tran3 = transforms.Grayscale(1)
tran1 = transforms.CenterCrop(256)
tran2 = transforms.ToTensor()
pic = tran2(tran1(tran3(pic)))
#pic = pic.resize((512,512),Image.BILINEAR)
pic = to_var(pic)

# pic = Normalize(pic)

# pic = np.transpose(pic,(2,0,1))
pic = pic.unsqueeze(0)

print("pic shape:{}".format(pic.shape))
# net = FPN([2, 4, 23, 3], 5)
net = resnet101(pretrained=False)
pthfile = r'./model/Best_MR2.pth'
net.load_state_dict(torch.load(pthfile))
net = net.eval()
softMax = nn.Softmax()
if torch.cuda.is_available():
    net.cuda()
    softMax.cuda()
out = net(pic)

#calculating parameters and flops 1
# flops, params = profile(net, inputs=(pic, ))
# flops, params = clever_format([flops, params], "%.3f")
# print(flops,params)

#calculating parameters and flops 2
# macs, params = get_model_complexity_info(net, (1, 256, 256), as_strings=True, print_per_layer_stat=True, verbose=True)
# print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
# print('{:<30}  {:<8}'.format('Number of parameters: ', params))

#visualization
plotout = torch.argmax(out, dim=1, keepdim=True)
plotout = plotout.squeeze()
# vis.surf(plotout.detach().cpu(), win='surfmap', opts=dict(title='surfmap'))
vis.heatmap(plotout.detach().cpu() , win="heatmap",opts=dict(title="heatmap"))

# _, c, _, _ = out.size()
#heatmap
# for i in range (c):
#     print('i = ', i)
#     plotout = out.squeeze()
#     plotout = plotout[i]
#     print("plotout:{}".format(plotout.shape))
#     vis.heatmap(plotout.detach().cpu(),win="heatmap{}".format(i),opts=dict(title="heatmap{}".format(i)))


out = out.data.cpu().numpy()
out = np.argmax(out,axis=1)
pre = decode_seg_map_sequence(out, plot=True)
save_image(pre,r'./testjpg.png')
