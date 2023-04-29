from glob import glob
import os
from PIL import Image
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torchvision import transforms
import torch.nn.functional as F
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")

# project imports
from .data_loader_cache import normalize, im_reader, im_preprocess
from .models import *

# Helpers
device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = None
hypar = {}  # paramters for inferencing


class GOSNormalize(object):
    '''
    Normalize the Image using torch.transforms
    '''

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        image = normalize(image, self.mean, self.std)
        return image


transform = transforms.Compose([GOSNormalize([0.5, 0.5, 0.5], [1.0, 1.0, 1.0])])


def load_model():
    # Set Parameters
    global hypar, net
    hypar["model_path"] = os.path.join(os.getcwd(), "model", "DIS")  # load trained weights from this path
    hypar["restore_model"] = "isnet-general-use.pth"  # name of the to-be-loaded weights
    hypar["interm_sup"] = False  # indicate if activate intermediate feature supervision

    #  choose floating point accuracy --
    hypar["model_digit"] = "full"  # indicates "half" or "full" accuracy of float number
    hypar["seed"] = 0

    hypar["cache_size"] = [1024, 1024]  # cached input spatial resolution, can be configured into different size

    # data augmentation parameters ---
    hypar["input_size"] = [1024,
                           1024]  # mdoel input spatial size, usually use the same value hypar["cache_size"], which means we don't further resize the images
    hypar["crop_size"] = [1024,
                          1024]  # random crop size from the input, it is usually set as smaller than hypar["cache_size"], e.g., [920,920] for data augmentation

    hypar["model"] = ISNetDIS()

    # Build Model
    net = build_model(hypar, device)
    net.eval()


def load_image(im_path, hypar):
    im = im_reader(im_path)
    if im.shape[-1] == 4:  # convert RGB
        im = im[:, :, :3]
    im, im_shp = im_preprocess(im, hypar["cache_size"])
    im = torch.divide(im, 255.0)
    shape = torch.from_numpy(np.array(im_shp))
    return transform(im).unsqueeze(0), shape.unsqueeze(0)  # make a batch of image, shape


def build_model(hypar, device):
    net = hypar["model"]  # GOSNETINC(3,1)

    # convert to half precision
    if (hypar["model_digit"] == "half"):
        net.half()
        for layer in net.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.float()

    net.to(device)

    if (hypar["restore_model"] != ""):
        net.load_state_dict(torch.load(hypar["model_path"] + "/" + hypar["restore_model"], map_location=device))
        net.to(device)
    net.eval()
    return net


def predict(net, inputs_val, shapes_val, hypar, device):
    '''
    Given an Image, predict the mask
    '''
    if net is None:
        load_model()
    if (hypar["model_digit"] == "full"):
        inputs_val = inputs_val.type(torch.FloatTensor)
    else:
        inputs_val = inputs_val.type(torch.HalfTensor)

    inputs_val_v = Variable(inputs_val, requires_grad=False).to(device)  # wrap inputs in Variable

    ds_val = net(inputs_val_v)[0]  # list of 6 results

    pred_val = ds_val[0][0, :, :, :]  # B x 1 x H x W    # we want the first one which is the most accurate prediction

    # recover the prediction spatial size to the orignal image size
    pred_val = torch.squeeze(
        F.upsample(torch.unsqueeze(pred_val, 0), (shapes_val[0][0], shapes_val[0][1]), mode='bilinear'))

    ma = torch.max(pred_val)
    mi = torch.min(pred_val)
    pred_val = (pred_val - mi) / (ma - mi)  # max = 1

    if device == 'cuda': torch.cuda.empty_cache()
    return (pred_val.detach().cpu().numpy() * 255).astype(np.uint8)  # it is the mask we need


def inference(in_path, out_path):
    """
    传入原始图像路径和处理后图像的存放路径
    :param in_path:
    :param out_path:
    :return:
    """
    load_model()
    image_path = in_path
    image_tensor, orig_size = load_image(image_path, hypar)
    mask = predict(net, image_tensor, orig_size, hypar, device)
    pil_mask = Image.fromarray(mask).convert('L')
    im_rgb = Image.open(in_path).convert("RGB")
    im_rgba = im_rgb.copy()
    im_rgba.putalpha(pil_mask)
    im_rgba.save(out_path)


def inference_dir(in_path, out_path):
    load_model()
    dataset_path = in_path.rstrip("/")  # Your dataset path
    result_path = out_path.rstrip("/")  # The folder path that you want to save the results
    # if not os.path.exists(result_path):
    #     os.mkdir(result_path)
    im_list = glob(dataset_path + "/*.jpg") + glob(dataset_path + "/*.JPG") + glob(dataset_path + "/*.jpeg") + glob(
        dataset_path + "/*.JPEG") + glob(dataset_path + "/*.png") + glob(dataset_path + "/*.PNG") + glob(
        dataset_path + "/*.bmp") + glob(dataset_path + "/*.BMP") + glob(dataset_path + "/*.tiff") + glob(
        dataset_path + "/*.TIFF")
    with torch.no_grad():
        for im_path in tqdm(im_list):
            if os.sep == "\\":  # windows
                im_name = im_path.split('\\')[-1].split('.')[0]
            else:  # linux
                im_name = im_path.split('/')[-1].split('.')[0]

            save_path = os.path.join(result_path, im_name + "_rmbg.png")
            inference(im_path, save_path)