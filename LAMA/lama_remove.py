import numpy as np
from pathlib import Path
from PIL import Image
from .lama_inpaint import inpaint_img_with_lama


def remove(img_path: str, save_path: str):
    """
    img_path: original image path
    save_path: 
    """
    prefix, _ = img_path.rsplit(".", 1)
    mask_path = prefix + "_expand.png"
    img = np.asarray(Image.open(img_path))
    mask = np.asarray(Image.open(mask_path))
    img_inpainted = inpaint_img_with_lama(img, mask, "./LAMA/lama/configs/prediction/default.yaml", "./model/big-lama")
    Image.fromarray(img_inpainted.astype(np.uint8)).save(save_path)
