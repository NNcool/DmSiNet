from matplotlib import pyplot as plt

from JNet import JNet
import torch.nn.functional as F
###############
# 模型代码没有问题，模型参数没有问题，那是啥子问题？

#https://github.com/jacobgil/pytorch-grad-cam 分割可解释性的github连接，本代码所使用的库文件
############
import torchvision.transforms as transforms
# warnings.filterwarnings('ignore')
# warnings.simplefilter('ignore')

import torch

import numpy as np

from PIL import Image

from models.DeepLab.DeeplabV3 import DeepLabV3
# from models.PSPNet.pspnet import PSPNet
from models.Unet.UNet import UNet
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam import GradCAM
from utils.utils import cvtColor, preprocess_input

#checkpoint_path = r'D:\onedrive\郑均\桌面\论文写作\4第四章语义分割模型设计\带有别人内容的代码\xingnengpingjia\models\Unet_best.pth'
#checkpoint_path = r'D:\onedrive\郑均\桌面\论文写作\4第四章语义分割模型设计\带有别人内容的代码\xingnengpingjia\models\PSPNet\PSP_best_epoch_weights.pth'



image_path = r"F:\lwl\JNET\img\c30_1_3.png"
# image_path = r"C:\0image_train\JPEGImages\a21_31_3_4.png"
class SemanticSegmentationTarget:
    def __init__(self, category, mask):
        self.category = category
        self.mask = torch.from_numpy(mask)
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()

    def __call__(self, model_output):
        return (model_output[self.category, :, :] * self.mask).sum()


def load_net(model,checkpoint_path):
      # 替换为你的权重文件的路径
    checkpoint = torch.load(checkpoint_path)
    # 加载权重
    model.load_state_dict(checkpoint)


# 包装函数，用来将图片转换成tensor
def f1(image_path):
    img = Image.open(image_path).convert('RGB')
    image = cvtColor(img)
    image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image, np.float32)), (2, 0, 1)), 0)
    images = torch.from_numpy(image_data)
    images = images.cuda()
    return images


# f2的功能是将模型输出的结果和图片的内容同时展示
def f2(output, image_path):
    image = np.array(Image.open(image_path))

    pr = F.softmax(output[0].permute(1, 2, 0), dim=-1).cpu().detach().numpy()

    pr = pr.argmax(axis=-1)
    binary_data = np.where(pr == 1, 255, 0)

    # 展示二值化后的图像
    plt.imshow(binary_data, cmap='gray', interpolation='nearest')
    plt.axis('off')
    plt.show()




def get_white_and_pic():
    model = UNet(3)
    # model = PSPNet(3,8,aux_branch=False)
   # model = JNet()
    model.cuda()
    load_net(model)
    model = model.eval()
    input_tensor = f1(image_path)

    output = model(input_tensor)
    f2(output, image_path)
def gram_cam(model,checkpoint_path,pic_path):

    #model = UNet(3)
    model.cuda()
    load_net(model,checkpoint_path)
    model = model.eval()
    input_tensor = f1(pic_path)
    output = model(input_tensor)

    image = np.array(Image.open(pic_path))
    rgb_img = np.float32(image) / 255

    image = np.array(Image.open(pic_path))
    normalized_masks = torch.nn.functional.softmax(output, dim=1).cpu()
    sem_classes = [
        "background", "red_pore", "blue_pore"
    ]
    sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(sem_classes)}

    car_category = sem_class_to_idx["blue_pore"]
    car_mask = normalized_masks[0, :, :, :].argmax(axis=0).detach().cpu().numpy()
    car_mask_uint8 = 255 * np.uint8(car_mask == car_category)
    car_mask_float = np.float32(car_mask == car_category)

    both_images = np.hstack((image, np.repeat(car_mask_uint8[:, :, None], 3, axis=-1)))
    img = Image.fromarray(both_images)

    target_layers = [model.classifier]
    targets = [SemanticSegmentationTarget(car_category, car_mask_float)]
    # use_cuda = torch.cuda.is_available()
    with GradCAM(model=model,
                 target_layers=target_layers,
                 ) as cam:
        grayscale_cam = cam(input_tensor=input_tensor,
                            targets=targets)[0, :]
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    img1 = Image.fromarray(cam_image)
    img1.show()
    img1.save("c_30JunNet_classifier.png")

if __name__ == '__main__':
    # pic_path = r"C:\0image_train\JPEGImages\a21_31_12_1.png"
    # pic_path = r"C:\0image_train\JPEGImages\n1_6_12.png"
    # pic_path = r"C:\0image_train\JPEGImages\n17_12_3.png"
    # pic_path = r"D:\onedrive\郑均\桌面\论文写作\4第四章语义分割模型设计\实验数据\标准图片\标准标注.png"
    pic_path = r"F:\lwl\JNET\img\c30_1_3.png"
    # ------------JunNet----------#
    checkpoint_path = r'F:\lwl\JNET\best_epoch_weights.pth'
    model = JNet(3)
    #
    # ----------------------#


    #------------UNet----------#
    # checkpoint_path = r'D:\onedrive\郑均\桌面\论文写作\4第四章语义分割模型设计\带有别人内容的代码\xingnengpingjia\models\Unet\Unet_best.pth'
    # checkpoint_path = r'D:\onedrive\郑均\桌面\论文写作\4第四章语义分割模型设计\带有别人内容的代码\xingnengpingjia\models\Unet\ep075-loss0.195-val_loss0.194.pth'
    # model = UNet(3)

    # ----------------------#

    #------------UNet----------#
    # checkpoint_path = r'D:\onedrive\郑均\桌面\论文写作\4第四章语义分割模型设计\带有别人内容的代码\xingnengpingjia\models\DeepLab\deeplab_best.pth'
    # model = DeepLabV3(3,3)
    #pic_path = r"D:\onedrive\郑均\桌面\论文写作\4第四章语义分割模型设计\实验数据\标准图片\标准标注.png"
    # ----------------------#

     #------------UNet----------#
    # checkpoint_path = r'D:\onedrive\郑均\桌面\论文写作\4第四章语义分割模型设计\带有别人内容的代码\xingnengpingjia\models\PSPNet\PSP_best_epoch_weights.pth'
    # model = PSPNet(3,8,aux_branch=False)
    # pic_path = r"D:\onedrive\郑均\桌面\论文写作\4第四章语义分割模型设计\实验数据\标准图片\标准标注.png"
    # ----------------------#

    gram_cam(model,checkpoint_path,pic_path)
    #get_white_and_pic()

