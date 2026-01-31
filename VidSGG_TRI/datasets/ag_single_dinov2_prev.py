"""
t frame 정보 받아서 직전 프레임(t-1)을 함께 return. img에 채널 축으로 concat해서 6채널 반환. 
target은 t frame의 것만 return
"""
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
import os
import torchvision.transforms.functional as F
from pathlib import Path

import torch
import torch.utils.data
from pycocotools import mask as coco_mask

from .torchvision_datasets import CocoDetection as TvCocoDetection
from util.misc import get_local_rank, get_local_size
# import datasets.transforms_single as T
import datasets.transforms_single_prev as T
from torch.utils.data.dataset import ConcatDataset
from .coco_video_parser import CocoVID

class DSGGDataset(TvCocoDetection):
    def __init__(self, img_folder, ann_file, transforms, cache_mode=False, local_rank=0, local_size=1, is_train=False):
        super(DSGGDataset, self).__init__(img_folder, ann_file,
                                            cache_mode=cache_mode, local_rank=local_rank, local_size=local_size)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(transforms)
        self.is_train = is_train
        self.img_folder = img_folder # PosixPath('data/action-genome')
        
        self.ann_file = ann_file # ag_train/test_coco_style.json
        self.cocovid = CocoVID(self.ann_file)
        
        print(' build DSGG dataset (returns t-1, t frames) ')

    def __getitem__(self, idx):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
            return single image tensor(img) and single target dict 
            현재 img_id로 파일 경로 읽고 파일명 파싱해서 t-1 프레임 경로 얻음
            두 이미지 리스트로 묶어 prepare(transform)에 전달
        """
        # (img_id = idx+1)
        coco = self.coco
        img_id = self.ids[idx]
        ann_ids = coco.getAnnIds(imgIds=img_id) # ex) [134944, 134945, 134946]
        target = coco.loadAnns(ann_ids)
        img_info = coco.loadImgs(img_id)[0] # ex) {'file_name': 'frames/0JA9E.mp4/000350.png', 'id': 2878, 'frame_id': 350, 'video_id': 117, 'width': 480, 'height': 270, 'is_vid_train_frame': True}
        
        video_id = img_info['video_id'] # ex) 117
        img_ids = self.cocovid.get_img_ids_from_vid(video_id) # video 내부 img ids
        
        # prev frame (t-1)
        if (img_id-1) in img_ids:
            prev_img_id = img_id-1
        else : 
            prev_img_id = img_id                        
        
        # load current frame (t)
        path_t = img_info['file_name'] # ex) 'frames/0JA9E.mp4/000350.png'        
        img_t = self.get_image(path_t)
        # load previous frame (t-1)
        prev_img_info = coco.loadImgs(prev_img_id)[0]
        path_prev = prev_img_info['file_name']
        img_prev = self.get_image(path_prev)
        

        image_id = img_id
        target = {'image_id': image_id, 'annotations': target}

        img, target = self.prepare([img_t, img_prev], target)

        if not self.is_train:
            target['img_path'] = path_t
        
        return img, target


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, transforms):
        self._transforms = transforms

    def __call__(self, images, target):
        # images: [img_t, img_prev] list 
        # image_t 크기 기준으로 target 처리

        w, h = images[0].size
        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]
        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0 or obj['iscrowd'] == -1]
        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)        
        
        attn_labels, spatial_labels, contacting_labels = [], [], []
        for oid in range(1, len(anno)):
            obj = anno[oid]
            attn_obj_label = torch.zeros(3)
            spatial_obj_label = torch.zeros(6)
            contacting_obj_label = torch.zeros(17)
            attn_obj_label[obj['attention_rel']] = 1
            spatial_obj_label[torch.tensor(obj['spatial_rel']) - 3] = 1
            contacting_obj_label[torch.tensor(obj['contact_rel']) - 9] = 1

            attn_labels.append(attn_obj_label)
            spatial_labels.append(spatial_obj_label)
            contacting_labels.append(contacting_obj_label)
        
        if len(attn_labels) > 0:
            attn_labels = torch.stack(attn_labels, dim=0)
            spatial_labels = torch.stack(spatial_labels, dim=0)
            contacting_labels = torch.stack(contacting_labels, dim=0)
        else:
            attn_labels = torch.zeros(0,3); spatial_labels = torch.zeros(0,6); contacting_labels = torch.zeros(0,17)
                
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])[keep]
        area = torch.tensor([obj["area"] for obj in anno])[keep]
        
        if len(keep) > 0:
            attn_labels = attn_labels[keep[1:]]
            spatial_labels = spatial_labels[keep[1:]]
            contacting_labels = contacting_labels[keep[1:]]

        target = {}
        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])
        target['boxes'] = boxes
        target['labels'] = classes
        target["iscrowd"] = iscrowd
        target["area"] = area

        images, target = self._transforms(images, target) # apply transforms
        
        if isinstance(images, list):
            image = torch.cat(images, dim=0) # (6, H, W)
        else : 
            image = images

        num_objs = len(boxes) - 1   
        if num_objs == 0 or (classes == 1).sum() == 0:
            target['obj_labels'] = torch.zeros((0,), dtype=torch.int64)
            target['sub_boxes'] = torch.zeros((0, 4), dtype=torch.float32)
            target['obj_boxes'] = torch.zeros((0, 4), dtype=torch.float32)
            target['attn_labels'] = torch.zeros((0, 3), dtype=torch.float32)
            target['spatial_labels'] = torch.zeros((0, 6), dtype=torch.float32)
            target['contacting_labels'] = torch.zeros((0, 17), dtype=torch.float32)
            target['matching_labels'] = torch.zeros((0,), dtype=torch.int64)
        else:
            target['obj_labels'] = target['labels'][1:]
            target['sub_boxes'] = target['boxes'][0].repeat((num_objs, 1))
            target['obj_boxes'] = target['boxes'][1:]
            target['attn_labels'] = attn_labels
            target['spatial_labels'] = spatial_labels
            target['contacting_labels'] = contacting_labels
            target['matching_labels'] = torch.ones_like(target['obj_labels'])

        return image, target
    
class ResizeFixed(object):
    def __init__(self, size_hw):  # (H, W)
        self.size_hw = size_hw

    def __call__(self, image, target):
        # image가 list로 들어오는 경우 ([img_t, img_prev]) 처리 추가
        h, w = self.size_hw
        
        if isinstance(image, list):
            image = [F.resize(im, [h, w]) for im in image]
        else:
            image = F.resize(image, [h, w])
        
        # boxes는 좌표 스케일링 (기존 로직 유지)
        if "boxes" in target:
            oh, ow = target["size"].tolist() if "size" in target else target["orig_size"].tolist()
            scale_y = h / float(oh)
            scale_x = w / float(ow)
            boxes = target["boxes"]
            boxes = boxes * torch.tensor([scale_x, scale_y, scale_x, scale_y], dtype=boxes.dtype)
            target["boxes"] = boxes
        
        target["size"] = torch.as_tensor([int(h), int(w)])
        return image, target

def make_coco_transforms(image_set):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # T.Normalize([0.4196, 0.3736, 0.3451], [0.2859, 0.2810, 0.2784])
    ])
    
    FIXED = 518 # dinov2 fixed size

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            # T.RandomResize([FIXED], max_size=FIXED),
            ResizeFixed((FIXED, FIXED)),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            # T.RandomResize([FIXED], max_size=FIXED),
            ResizeFixed((FIXED, FIXED)),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.ag_path)
    assert root.exists(), f'provided Action Genome path {root} does not exist'
    mode = 'instances'
    PATHS = {
        "train": (root, root / "annotations" / 'ag_train_coco_style.json'),
        "val": (root, root / "annotations" / 'ag_test_coco_style.json'),
    }
    img_folder, anno_file = PATHS[image_set]
    dataset = DSGGDataset(img_folder, anno_file, transforms=make_coco_transforms(image_set), cache_mode=args.cache_mode, local_rank=get_local_rank(), local_size=get_local_size(), is_train=(not args.eval))
    return dataset
