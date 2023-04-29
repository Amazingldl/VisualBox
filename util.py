import os
import numpy as np
import torch
from PIL import Image
from typing import Literal
from RealESRGAN import RealESRGAN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def real_esrgan(img_path, save_path, scale):
    """
    建议保存图片为jpg格式，若为png格式则存储空间占用会很大
    :param img_path:
    :param save_path:
    :param scale:
    :return:
    """
    model = RealESRGAN(device, scale=scale)
    if scale == 2:
        model.load_weights('./model/RealESRGAN/RealESRGAN_x2plus.pth', download=False)
    else:
        model.load_weights('./model/RealESRGAN/RealESRGAN_x4plus.pth', download=False)
    image = Image.open(img_path).convert('RGB')
    sr_image = model.predict(image)
    sr_image.save(save_path)


def is_imagefile(path):
    if (path.endswith("jpg") or path.endswith("JPG")
            or path.endswith("png") or path.endswith("PNG")
            or path.endswith("jpeg") or path.endswith("JPEG")
            or path.endswith("bmp") or path.endswith("BMP")
            or path.endswith("tiff") or path.endswith("TIFF")):
        return True
    else:
        return False


def is_imagefile_exist(path):
    if os.path.exists(path) and is_imagefile(path):
        return True
    else:
        return False


def set_proxy():
    os.environ['http_proxy'] = "http://127.0.0.1:7890"
    os.environ['https_proxy'] = "http://127.0.0.1:7890"


