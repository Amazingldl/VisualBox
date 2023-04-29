import numpy as np
from DIS import dis_inference
from DIS import dis_inference_dir
from SAM import sam_inference
from SD import hf_remove
from LAMA import lama_remove
from SD import StableDiffusionModel
from SD import StableDiffusionControlNetModel
from OF import of_inference

from fastapi import FastAPI
from util import *
from typing import List, Dict, Optional, Literal

Point = List[List[int]]
Label = List[int]

app = FastAPI()


# uvicorn all_api:app --reload --port 6006
@app.get("/")
def welcome():
    return "Welcome!"


@app.post("/api/tools/rmbg")
def rm_bg(
        in_path: str,
        out_path: str,
) -> Dict:
    """
    移除背景接口
    :param in_path: str
    :param out_path: str
    :return: Dict
    """
    if is_imagefile_exist(in_path) and is_imagefile(out_path):
        dis_inference(in_path, out_path)
        return {"result": "success"}
    else:
        return {"result": "failed, wrong path"}


@app.post("/api/tools/rmbgdir")
def rm_bg_dir(
        in_path: str,
        out_path: str,
) -> Dict:
    """
    批量移除背景接口
    :param in_path: str
    :param out_path: str
    :return: Dict
    """
    if os.path.exists(in_path) and os.path.exists(out_path):
        dis_inference_dir(in_path, out_path)
        return {"result": "success"}
    else:
        return {"result": "failed, wrong path"}


@app.post("/api/tools/sam")
def sam(in_path: str,
        points: Point,  # 200,450
        labels: Label,
        box: Optional[Label] = None,
        ) -> Dict:
    """
    传入：
      一张图片路径
      操作点坐标（可多个）
      操作点类型
      框坐标（最多一个）
    产出：
      分割图：路径为"原路径"但图片以_seg.png结尾（此处可以先返回） 分割图对程序其实没有作用，只是展示给用户看的
      maks图：路径为"原路径"但图片以_mask.png结尾
      expand图：路径为"原路径"但图片以_expand.png结尾
    返回：
      只返回状态，前端自动寻找分割图的路径
    """
    print(points, labels)
    if is_imagefile_exist(in_path):
        if len(points) != len(labels):
            return {"result": "failed, wrong parameter"}
        points = np.array(points)
        labels = np.array(labels)
        if box is not None:
            box = np.array(box)
        sam_inference(in_path, points, labels, box)
        return {"result": "success"}
    else:
        return {"result": "failed, wrong path"}


@app.post("api/tools/of")
def seg_oneformer(in_path, out_path, out_map_path):
    """
    oneformer分割接口
    :param in_path: str # 原始图像路径
    :param out_path: str    # 分割图（带有物体标号）的保存路径
    :param out_map_path: str    # 分割图（无标号）的保存路径
    :return: Dict
    """
    if is_imagefile_exist(in_path) and is_imagefile(out_path) and is_imagefile(out_map_path):
        out, out_map = of_inference(in_path)
        out.save(out_path)
        out_map.save(out_map_path)
        return {"result": "success"}
    else:
        return {"result": "failed, wrong path"}


@app.post("/api/tools/remove_sd")
def remove_sd(
        in_path: str,
        out_path: str,
) -> Dict:
    """
    移除对象接口--使用stable diffusion
    :param in_path: str
    :param out_path: str
    :return: Dict
    """
    prefix, _ = in_path.rsplit(".", 1)
    if not is_imagefile_exist(in_path):
        return {"result": "failed, wrong path"}
    elif not os.path.exists(prefix + "_expand.png"):
        return {"result": "failed, please generate mask image first"}
    else:
        hf_remove(in_path, out_path)
        return {"result": "success"}


@app.post("/api/tools/remove_lama")
def remove_lama(
        in_path: str,
        out_path: str,
) -> Dict:
    """
    移除对象接口--使用lama
    :param in_path: str
    :param out_path: str
    :return: Dict
    """
    prefix, _ = in_path.rsplit(".", 1)
    if not is_imagefile_exist(in_path):
        return {"result": "failed, wrong path"}
    elif not os.path.exists(prefix + "_expand.png"):
        return {"result": "failed, please generate mask image first"}
    else:
        lama_remove(in_path, out_path)
        return {"result": "success"}


@app.post("/api/tools/text2img")
def text2img(
        save_path: str,
        prompt: str,
        n_prompt: Optional[str] = None,
        width: Optional[int] = 1024,
        height: Optional[int] = 576,
) -> Dict:
    """
    文字生成图片接口
    :param save_path: str
    :param prompt: str
    :param n_prompt: Optional[str]
    :param width: Optional[int]
    :param height: Optional[int]
    :return: Dict
    """
    if is_imagefile(save_path):
        sd = StableDiffusionModel()
        sd.text2img(save_path, prompt, n_prompt, width, height)
        return {"result": "success"}
    else:
        return {"result": "failed, wrong path"}


@app.post("/api/tools/img2img")
def img2img(
        in_path: str,
        save_path: str,
        prompt: str,
        n_prompt: Optional[str] = None,
) -> Dict:
    """
    图片生成图片接口
    :param in_path: str
    :param save_path: str
    :param prompt: str
    :param n_prompt: Optional[str]
    :return: Dict
    """
    if is_imagefile_exist(in_path) and is_imagefile(save_path):
        sd = StableDiffusionModel()
        sd.img2img(in_path, save_path, prompt, n_prompt)
        return {"result": "success"}
    else:
        return {"result": "failed, wrong path"}


@app.post("/api/tools/text2img_seg")
def text2img_seg(
        control_path: str,
        save_path: str,
        prompt: str,
        n_prompt: Optional[str] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        save_seg: Optional[bool] = False,
        seg_path: Optional[str] = None,
) -> Dict:
    """
    ControlNet-Seg接口
    :param control_path: str
    :param save_path: str
    :param prompt: str
    :param n_prompt: Optional[str] = None
    :param save_seg: Optional[bool] = False
    :param seg_path: Optional[str] = None
    :return: Dict
    """
    if is_imagefile_exist(control_path) and is_imagefile(save_path) and (
            save_seg is True and is_imagefile(seg_path) or save_seg is False and seg_path is None):
        sdc = StableDiffusionControlNetModel()
        sdc.text2img_seg(control_path, save_path, prompt, n_prompt=n_prompt, height=height, width=width, save_seg=save_seg, seg_path=seg_path)
        return {"result": "success"}
    else:
        return {"result": "failed, wrong path"}


@app.post("/api/tools/attach")
def attach():
    pass


@app.post("/api/tools/super_reso")
def super_reso(in_path: str, out_path: str, scale: Literal['2', '4'] = '2'):
    scale = int(scale)
    if is_imagefile_exist(in_path) and is_imagefile(out_path):
        real_esrgan(in_path, out_path, scale)
        return {"result": "success"}
    else:
        return {"result": "failed, wrong path"}
