import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor
from torchvision import transforms


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='.', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='.', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


def save_seg(img, save_path, mask, input_point=None, input_label=None, box=None):
    """
    :param img: ndarray
    :param save_path:
    :param mask:
    :param input_point:
    :param input_label:
    :param box:
    :return:
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    show_points(input_point, input_label, plt.gca())
    show_mask(mask, plt.gca())
    if input_point is not None:
        show_points(input_point, input_label, plt.gca())
    if box is not None:
        show_box(box, plt.gca())
    plt.axis('off')
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)


def save_mask(masks, save_path):
    """
    :param masks:
    :param save_path:
    :return:
    """
    mask = masks[0]
    mask = mask.astype(np.uint8) * 255
    mask_img = Image.fromarray(mask)
    mask_img.save(save_path)

    # for mask in masks:
    #     h, w = mask.shape[0], mask.shape[1]
    #     mask_img = Image.new('L', (w, h), 0)  # 灰度图，所有mask图像都是单通道的灰度图
    #     for i in range(h):
    #         for j in range(w):
    #             if mask[i][j]:
    #                 mask_img.putpixel((j, i), 255)
    #             else:
    #                 mask_img.putpixel((j, i), 0)
    # mask_img.save(save_path)


def save_expand(masks, save_path, dilate_factor=15):
    """
    :param masks: ndarray (HxW)
    :param save_path:
    :param dilate_factor:
    :return:
    """
    mask = masks[0].astype(np.uint8) * 255
    dilate_mask = cv2.dilate(
        mask,
        np.ones((dilate_factor, dilate_factor), np.uint8),
        iterations=1
    )
    dilate_img = Image.fromarray(dilate_mask)
    dilate_img.save(save_path)

    # img = mask_img
    # img_tensor = transforms.ToTensor()(img)
    # dilation_kernel = torch.ones(kernel_num, kernel_num)
    # padding_num = (kernel_num - 1) // 2
    # dilated_img_tensor = torch.nn.functional.conv2d(
    #     torch.nn.functional.pad(img_tensor.unsqueeze(0), (padding_num,) * 4),
    #     dilation_kernel.unsqueeze(0).unsqueeze(0),
    # ).squeeze(0)
    # dilated_img = transforms.ToPILImage()(dilated_img_tensor.cpu())
    # dilated_img = binarize_image(dilated_img, threshold_pixel)
    # dilated_img.save(save_path)


sam_checkpoint = "./model/SAM/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)


def inference(img_path, input_points, input_labels, input_box=None):
    """
    :param img_path:
    :param input_points: ndarray
    :param input_labels: ndarray
    :param input_box: ndarray
    :return:
    """
    prefix, _ = img_path.rsplit(".", 1)
    seg_path = prefix + "_seg.png"
    mask_path = prefix + "_mask.png"
    expand_path = prefix + "_expand.png"

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    predictor.set_image(img)
    masks, scores, _ = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        box=input_box,
        multimask_output=False,
    )

    # 1. 存储seg image
    save_seg(img, seg_path, masks, input_points, input_labels, input_box)
    # 2. 存储mask image
    save_mask(masks, mask_path)
    # 3. 存储expand image
    save_expand(masks, expand_path)
