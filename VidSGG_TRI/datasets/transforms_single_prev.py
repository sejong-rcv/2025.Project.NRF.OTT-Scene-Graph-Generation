# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Transforms and data augmentation for both image + bbox.
"""
import random

import PIL
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F

from util.box_ops import box_xyxy_to_cxcywh
from util.misc import interpolate


def crop(image, target, region):
    cropped_image = F.crop(image, *region)

    target = target.copy()
    i, j, h, w = region

    # should we do something wrt the original size?
    target["size"] = torch.tensor([h, w])

    fields = ["labels", "area", "iscrowd"]

    if "boxes" in target:
        boxes = target["boxes"]
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
        target["boxes"] = cropped_boxes.reshape(-1, 4)
        target["area"] = area
        fields.append("boxes")

    if "masks" in target:
        # FIXME should we update the area here if there are no boxes?
        target['masks'] = target['masks'][:, i:i + h, j:j + w]
        fields.append("masks")

    # remove elements for which the boxes or masks that have zero area
    if "boxes" in target or "masks" in target:
        # favor boxes selection when defining which elements to keep
        # this is compatible with previous implementation
        if "boxes" in target:
            cropped_boxes = target['boxes'].reshape(-1, 2, 2)
            keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
        else:
            keep = target['masks'].flatten(1).any(1)

        for field in fields:
            target[field] = target[field][keep]

    return cropped_image, target


def hflip(image, target):
    flipped_image = F.hflip(image)

    w, h = image.size

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
        target["boxes"] = boxes

    if "masks" in target:
        target['masks'] = target['masks'].flip(-1)

    return flipped_image, target


def resize(image, target, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image.size, size, max_size)
    rescaled_image = F.resize(image, size)

    if target is None:
        return rescaled_image, None

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["boxes"] = scaled_boxes

    if "area" in target:
        area = target["area"]
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area

    h, w = size
    target["size"] = torch.tensor([h, w])

    if "masks" in target:
        target['masks'] = interpolate(
            target['masks'][:, None].float(), size, mode="nearest")[:, 0] > 0.5

    return rescaled_image, target


def pad(image, target, padding):
    # assumes that we only pad on the bottom right corners
    padded_image = F.pad(image, (0, 0, padding[0], padding[1]))
    if target is None:
        return padded_image, None
    target = target.copy()
    # should we do something wrt the original size?
    target["size"] = torch.tensor(padded_image[::-1])
    if "masks" in target:
        target['masks'] = torch.nn.functional.pad(target['masks'], (0, padding[0], 0, padding[1]))
    return padded_image, target


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        region = T.RandomCrop.get_params(img, self.size)
        return crop(img, target, region)


class RandomSizeCrop(object):
    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img: PIL.Image.Image, target: dict):
        w = random.randint(self.min_size, min(img.width, self.max_size))
        h = random.randint(self.min_size, min(img.height, self.max_size))
        region = T.RandomCrop.get_params(img, [h, w])
        return crop(img, target, region)


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        image_width, image_height = img.size
        crop_height, crop_width = self.size
        crop_top = int(round((image_height - crop_height) / 2.))
        crop_left = int(round((image_width - crop_width) / 2.))
        return crop(img, target, (crop_top, crop_left, crop_height, crop_width))


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        # img가 list인 경우 ([t, t-1])
        if isinstance(img, list):
            if random.random() < self.p:
                # 동일하게 둘 다 Flip
                img = [F.hflip(im) for im in img]
                if "boxes" in target:
                    w, h = img[0].size
                    boxes = target["boxes"]
                    boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
                    target["boxes"] = boxes
            return img, target
        
        # 기존 단일 이미지 로직
        if random.random() < self.p:
            return hflip(img, target)
        return img, target
    

class RandomHorizontalFlip_targets(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target, target_prev=None):
        if random.random() >= self.p:
            return img, target, target_prev

        if isinstance(img, list):
            img = [F.hflip(im) for im in img]
            # t target
            if target is not None and "boxes" in target:
                w, h = img[0].size
                boxes = target["boxes"]
                boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
                target["boxes"] = boxes
            # t-1 target
            if target_prev is not None and "boxes" in target_prev:
                w, h = img[1].size
                boxes = target_prev["boxes"]
                boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
                target_prev["boxes"] = boxes

            return img, target, target_prev
        # 단일 이미지 경로(기존 호환)
        img, target = hflip(img, target)
        return img, target, target_prev
    



class RandomResize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, target=None):
        size = random.choice(self.sizes)
        return resize(img, target, size, self.max_size)


class RandomPad(object):
    def __init__(self, max_pad):
        self.max_pad = max_pad

    def __call__(self, img, target):
        pad_x = random.randint(0, self.max_pad)
        pad_y = random.randint(0, self.max_pad)
        return pad(img, target, (pad_x, pad_y))


class RandomSelect(object):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """
    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return self.transforms1(img, target)
        return self.transforms2(img, target)

# ToTensor: 리스트 처리
class ToTensor(object):
    def __call__(self, img, target):
        if isinstance(img, list):
            return [F.to_tensor(im) for im in img], target
        return F.to_tensor(img), target

class ToTensor_targets(object):
    def __call__(self, img, target, target_prev=None):
        if isinstance(img, list):
            img = [F.to_tensor(im) for im in img]
        else:
            img = F.to_tensor(img)
        return img, target, target_prev
    
class RandomErasing(object):

    def __init__(self, *args, **kwargs):
        self.eraser = T.RandomErasing(*args, **kwargs)

    def __call__(self, img, target):
        return self.eraser(img), target

#  Normalize: 리스트 처리
class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        if isinstance(image, list):
            image = [F.normalize(im, mean=self.mean, std=self.std) for im in image]
        else:
            image = F.normalize(image, mean=self.mean, std=self.std)
            
        if target is None: return image, None
        target = target.copy()
        
        # ... (Box Coordinate Normalize 로직 기존 동일) ...
        # 주의: image가 list면 image[0].shape 사용
        ref_img = image[0] if isinstance(image, list) else image
        h, w = ref_img.shape[-2:]
        if "boxes" in target:
            boxes = target["boxes"]
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["boxes"] = boxes
        return image, target
    
class Normalize_targets(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def _norm_target_boxes(self, target, ref_img_tensor):
        if target is None:
            return None
        target = target.copy()
        h, w = ref_img_tensor.shape[-2:]
        if "boxes" in target:
            boxes = target["boxes"]
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["boxes"] = boxes
        return target

    def __call__(self, image, target=None, target_prev=None):
        if isinstance(image, list):
            image = [F.normalize(im, mean=self.mean, std=self.std) for im in image]
            target = self._norm_target_boxes(target, image[0])
            target_prev = self._norm_target_boxes(target_prev, image[1] if target_prev is not None else image[0])
            return image, target, target_prev

        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image, None, target_prev
        target = self._norm_target_boxes(target, image)
        return image, target, target_prev

    

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string

# return both (t-1, t) target
class Compose_targets(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target, target_prev=None):
        for t in self.transforms:
            image, target, target_prev = t(image, target, target_prev)
        if target_prev is None:
            return image, target
        return image, target, target_prev


class ResizeFixed(object):
    def __init__(self, size_hw):  # (H, W) 튜플
        self.size_hw = size_hw

    def __call__(self, image, target):
        h, w = self.size_hw
        if isinstance(image, list): # # image가 list로 들어오는 경우 ([img_t, img_prev]) 처리 추가
            image = [F.resize(im, [h, w]) for im in image] # 리스트 내 모든 이미지를 resize
        else:
            image = F.resize(image, [h, w]) # 단일 이미지인 경우
        
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
    

class ResizeFixed_targets(object):
    def __init__(self, size_hw):  # (H, W)
        self.size_hw = size_hw

    def _resize_one_target(self, target, new_h, new_w):
        if target is None:
            return None
        target = target.copy()

        # old size
        if "size" in target:
            old_h, old_w = target["size"].tolist()
        else:
            old_h, old_w = target["orig_size"].tolist()

        scale_y = new_h / float(old_h)
        scale_x = new_w / float(old_w)

        if "boxes" in target:
            boxes = target["boxes"]
            target["boxes"] = boxes * torch.tensor(
                [scale_x, scale_y, scale_x, scale_y],
                dtype=boxes.dtype,
                device=boxes.device,
            )

        if "area" in target:
            area = target["area"]
            target["area"] = area * (scale_x * scale_y)

        target["size"] = torch.as_tensor([int(new_h), int(new_w)], dtype=torch.int64)
        return target

    def __call__(self, image, target, target_prev=None):
        new_h, new_w = self.size_hw

        # resize images (PIL 단계에서 수행 중이므로 F.resize 사용 가능)
        if isinstance(image, list):
            image = [F.resize(im, [new_h, new_w]) for im in image]
        else:
            image = F.resize(image, [new_h, new_w])

        # resize both targets independently
        target = self._resize_one_target(target, new_h, new_w)
        target_prev = self._resize_one_target(target_prev, new_h, new_w)

        return image, target, target_prev
