import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from PIL import Image
from typing import Literal
from RealESRGAN import RealESRGAN
from diffusers import StableDiffusionInpaintPipeline, ControlNetModel, UniPCMultistepScheduler
from LAMA import inpaint_img_with_lama
from typing import List, Dict, Optional, Literal
from SD import StableDiffusionModel
from SD import StableDiffusionControlNetModel



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
    
    if isinstance(img_path, str):
        image = Image.open(img_path).convert('RGB')
    elif isinstance(img_path, np.ndarray):
        image = Image.fromarray(img_path).convert('RGB')
    sr_image = model.predict(image)
    if save_path is not None:
        sr_image.save(save_path)
    else:
        return sr_image


def real_esrgan_app(img, scale):
    model = RealESRGAN(device, scale=scale)
    if scale == 2:
        model.load_weights('./model/RealESRGAN/RealESRGAN_x2plus.pth', download=False)
    else:
        model.load_weights('./model/RealESRGAN/RealESRGAN_x4plus.pth', download=False)

    image = Image.fromarray(img).convert('RGB')
    sr_image = model.predict(image)
    return sr_image


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


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


def plt_seg(img, mask, input_point=None, input_label=None, box=None):
    dpi = plt.rcParams['figure.dpi']
    height, width = img.shape[:2]
    plt.figure(figsize=(width / dpi / 0.77, height / dpi / 0.77))
    # plt.figure(figsize=(10, 10))
    plt.imshow(img)
    show_mask(mask, plt.gca())
    if input_point is not None:
        show_points(input_point, input_label, plt.gca())
    if box is not None:
        show_box(box, plt.gca())
    plt.axis('off')
    plt.savefig("temp.png", bbox_inches="tight", pad_inches=0)
    img = cv2.imread("temp.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def gene_mask(masks):
    return masks[0].astype(np.uint8) * 255


def gene_expand(mask_img):
    mask = mask_img.astype(np.uint8)
    dilate_factor = 20
    dilate_mask = cv2.dilate(
        mask,
        np.ones((dilate_factor, dilate_factor), np.uint8),
        iterations=1
    )
    return dilate_mask


def gene_sd_inpaint(prompt, image, mask_image):
    # prompt = "background"
    # controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "./model/SD/stable-diffusion-inpainting",
        # "./model/HF/stable-diffusion-2-inpainting",
        # controlnet=controlnet,
        torch_dtype=torch.float16,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()
    pipe.to("cuda")

    # prepare canny image
    # canny_image = cv2.Canny(image, 100, 200)
    # canny_image = canny_image[:, :, None]
    # canny_image = np.concatenate([canny_image, canny_image, canny_image], axis=2)

    h, w = image.shape[:2]  # image.shape = (H, W, 3)   mode=RGB
    h = h//8 * 8
    w = w//8 * 8

    # image = Image.fromarray(image).resize((512, 512))
    # mask_image = Image.fromarray(mask_image).resize((512, 512))
    image = Image.fromarray(image)
    mask_image = Image.fromarray(mask_image)
    # canny_image = Image.fromarray(canny_image).resize((512, 512))

    removed_img = pipe(
        prompt,
        image=image,
        mask_image=mask_image,
        guidance_scale=7.5,
        height=h,
        width=w,
        # control_image=canny_image,
    ).images[0]
    # return removed_img.resize((w, h))
    return removed_img


def gene_lama_removed(image, mask_image):
    img_inpainted = inpaint_img_with_lama(image, mask_image, "./LAMA/lama/configs/prediction/default.yaml", "./model/big-lama")
    return img_inpainted

def text2img(style, prompt, n_prompt, width, height, seed, img_nums):
    styles = {"传统中式": "Chinese, Traditional Chinese interior design",
              "新中式": "New Chinese interior design",
              "日式侘寂风": "chaji, wabi-sabi interior design",
              "北欧清新风": "North European interior design",
              "现代极简风": "Modern interior design",
              }
    if style:
        prompt = styles[style] + ", " + prompt
    sd = StableDiffusionModel()
    print(prompt, n_prompt, seed, width, height, img_nums)
    img = sd.text2img(None, prompt, n_prompt, width, height, seed, int(img_nums))
    return img

def img2img(style, img, prompt, n_prompt, seed, img_nums):
    styles = {"传统中式": "Chinese, Traditional Chinese interior design",
              "新中式": "New Chinese interior design",
              "日式侘寂风": "chaji, wabi-sabi interior design",
              "北欧清新风": "North European interior design",
              "现代极简风": "Modern interior design",
              }
    if style:
        prompt = styles[style] + ", " + prompt
    sd = StableDiffusionModel()
    print(prompt, n_prompt, seed, img_nums)
    img = sd.img2img(img, None, prompt, n_prompt, seed, int(img_nums))
    return img