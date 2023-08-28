###########################################################################
# Created by: NTU
# Email: heshuting555@gmail.com
# Copyright (c) 2023
###########################################################################


import json
import logging
import numpy as np
import os

from detectron2.structures import Boxes, BoxMode, PolygonMasks
from detectron2.data import DatasetCatalog, MetadataCatalog
from tqdm import tqdm
"""
This file contains functions to parse MeViS dataset of
COCO-format annotations into dicts in "Detectron2 format".
"""

logger = logging.getLogger(__name__)

__all__ = ["load_mevis_json", "register_mevis_instances"]


def load_mevis_json(image_root, json_file):

    num_instances_without_valid_segmentation = 0
    num_instances_valid_segmentation = 0


    ann_file = json_file
    with open(str(ann_file), 'r') as f:
        subset_expressions_by_video = json.load(f)['videos']
    videos = list(subset_expressions_by_video.keys())
    print('number of video in the datasets:{}'.format(len(videos)))
    metas = []
    if image_root.split('/')[-1] == 'train':
        mask_json = os.path.join(image_root, 'mask_dict.json')
        print(f'Loading masks form {mask_json} ...')
        with open(mask_json) as fp:
            mask_dict = json.load(fp)

        for vid in videos:
            vid_data = subset_expressions_by_video[vid]
            vid_frames = sorted(vid_data['frames'])
            vid_len = len(vid_frames)
            if vid_len < 2:
                continue
            for exp_id, exp_dict in vid_data['expressions'].items():
                meta = {}
                meta['video'] = vid
                meta['exp'] = exp_dict['exp']
                meta['obj_id'] = [int(x) for x in exp_dict['obj_id']]
                meta['anno_id'] = [str(x) for x in exp_dict['anno_id']]
                meta['frames'] = vid_frames
                meta['exp_id'] = exp_id
                meta['category'] = 0
                meta['length'] = vid_len
                metas.append(meta)
    else:
        for vid in videos:
            vid_data = subset_expressions_by_video[vid]
            vid_frames = sorted(vid_data['frames'])
            vid_len = len(vid_frames)
            for exp_id, exp_dict in vid_data['expressions'].items():
                meta = {}
                meta['video'] = vid
                meta['exp'] = exp_dict['exp']
                meta['obj_id'] = -1
                meta['anno_id'] = -1
                meta['frames'] = vid_frames
                meta['exp_id'] = exp_id
                meta['category'] = 0
                meta['length'] = vid_len
                metas.append(meta)

    dataset_dicts = []
    for vid_dict in tqdm(metas):
        record = {}
        record["file_names"] = [os.path.join(image_root, 'JPEGImages', vid_dict['video'], vid_dict["frames"][i]+ '.jpg') for i in range(vid_dict["length"])]
        record["length"] = vid_dict["length"]
        video_name, exp, anno_ids, obj_ids, category, exp_id = \
            vid_dict['video'], vid_dict['exp'], vid_dict['anno_id'], vid_dict['obj_id'], vid_dict['category'],  vid_dict['exp_id']

        exp = " ".join(exp.lower().split())
        if "eval_idx" in vid_dict:
            record["eval_idx"] = vid_dict["eval_idx"]

        video_objs = []
        if image_root.split('/')[-1] == 'train':
            for frame_idx in range(record["length"]):
                frame_objs = []
                for x, obj_id in zip(anno_ids, obj_ids):
                    obj = {}
                    segm = mask_dict[x][frame_idx]
                    if not segm:
                        num_instances_without_valid_segmentation += 1
                        continue
                    num_instances_valid_segmentation += 1
                    bbox = [0, 0, 0, 0]
                    obj["id"] = obj_id
                    obj["segmentation"] = segm
                    obj["category_id"] = category
                    obj["bbox"] = bbox
                    obj["bbox_mode"] = BoxMode.XYXY_ABS
                    frame_objs.append(obj)
                video_objs.append(frame_objs)
        record["annotations"] = video_objs
        record["sentence"] = exp
        record["exp_id"] = exp_id
        record["video_name"] = video_name
        dataset_dicts.append(record)

    if num_instances_without_valid_segmentation > 0:
        logger.warning(
            "Total {} instance and Filtered out {} instances without valid segmentation. ".format(
                num_instances_valid_segmentation, num_instances_without_valid_segmentation
            )
        )
    return dataset_dicts


def bounding_box(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax # y1, y2, x1, x2


def register_mevis_instances(name, json_file, image_root):
    """
    Register a dataset in YTVIS's json annotation format for
    instance tracking.

    Args:
        name (str): the name that identifies a dataset, e.g. "ytvis_train".
        metadata (dict): extra metadata associated with this dataset.  You can
            leave it as an empty dict.
        json_file (str): path to the json instance annotation file.
        image_root (str or path-like): directory which contains all the images.
    """
    assert isinstance(name, str), name
    assert isinstance(json_file, (str, os.PathLike)), json_file
    assert isinstance(image_root, (str, os.PathLike)), image_root
    # 1. register a function which returns dicts
    DatasetCatalog.register(name, lambda: load_mevis_json(image_root, json_file))

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, evaluator_type="ytvis")


if __name__ == "__main__":
    """
    Test the YTVIS json dataset loader.
    """
    import detectron2.data.datasets  # noqa # add pre-defined metadata

    image_root = "./datasets/mevis/valid_u"
    json_file = "./datasets/mevis/valid_u/meta_expressions.json"
    dicts = load_mevis_json(image_root, json_file)
    print("Done loading {} samples.".format(len(dicts)))
