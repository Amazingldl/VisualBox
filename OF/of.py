import os
import sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.append(os.path.dirname(__file__))  # 用来解决包的引用问题
current_path = os.path.dirname(__file__)    # 用来解决文件相对路径的问题

import imutils
import cv2
from PIL import Image
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.data import MetadataCatalog

from oneformer import (
    add_oneformer_config,
    add_common_config,
    add_swin_config,
    add_dinat_config,
)

from oneformer.demo.defaults import DefaultPredictor
from oneformer.demo.visualizer import Visualizer, ColorMode


configs = {
    "coco": {
        "name": "150_16_swin_l_oneformer_coco_100ep.pth",
        "config": os.path.join(current_path, 'configs/coco/oneformer_swin_large_IN21k_384_bs16_100ep.yaml'),
    },
    "ade20k": {
        "name": "250_16_swin_l_oneformer_ade20k_160k.pth",
        "config": os.path.join(current_path, 'configs/ade20k/oneformer_swin_large_IN21k_384_bs16_160k.yaml')
    }
}


def load_model(model_name):
    config = configs[model_name]["config"]
    modelpath = os.path.join("model", "OF", configs[model_name]["name"])
    model, metadata = make_detectron2_model(config, modelpath)
    return model, metadata


def make_detectron2_model(config_path, ckpt_path):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_common_config(cfg)
    add_swin_config(cfg)
    add_oneformer_config(cfg)
    add_dinat_config(cfg)
    cfg.merge_from_file(config_path)
    cfg.MODEL.WEIGHTS = ckpt_path
    cfg.freeze()
    metadata = MetadataCatalog.get(cfg.DATASETS.TEST_PANOPTIC[0] if len(cfg.DATASETS.TEST_PANOPTIC) else "__unused")
    return DefaultPredictor(cfg), metadata


def semantic_run(img, predictor, metadata):
    visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, instance_mode=ColorMode.IMAGE)
    predictions = predictor(img[:, :, ::-1], "semantic")  # Predictor of OneFormer must use BGR image !!!
    out = visualizer.draw_sem_seg(predictions["sem_seg"].argmax(dim=0).cpu(), alpha=0.5).get_image()
    visualizer_map = Visualizer(img, is_img=False, metadata=metadata, instance_mode=ColorMode.IMAGE)
    out_map = visualizer_map.draw_sem_seg(predictions["sem_seg"].argmax(dim=0).cpu(), alpha=1, is_text=False).get_image()
    return out, out_map


def inference(img_path):
    img = cv2.imread(img_path)
    img = imutils.resize(img, width=640)
    model, metadata = load_model("ade20k")
    model.model.to("cuda")
    out, out_map = semantic_run(img, model, metadata)
    out = Image.fromarray(out)
    out_map = Image.fromarray(out_map)
    return out, out_map
