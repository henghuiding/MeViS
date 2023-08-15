import copy
import numpy as np
import logging
import sys
from fvcore.transforms.transform import (
    HFlipTransform,
    NoOpTransform,
    VFlipTransform,
)
from PIL import Image
from typing import Tuple
from fvcore.transforms.transform import (
    BlendTransform,
    CropTransform,
    HFlipTransform,
    NoOpTransform,
    PadTransform,
    Transform,
    TransformList,
    VFlipTransform,
)

from detectron2.data import transforms as T


class RandomApplyClip(T.Augmentation):
    """
    Randomly apply an augmentation with a given probability.
    """

    def __init__(self, tfm_or_aug, prob=0.5, clip_frame_cnt=1):
        """
        Args:
            tfm_or_aug (Transform, Augmentation): the transform or augmentation
                to be applied. It can either be a `Transform` or `Augmentation`
                instance.
            prob (float): probability between 0.0 and 1.0 that
                the wrapper transformation is applied
        """
        super().__init__()
        self.aug = T.augmentation._transform_to_aug(tfm_or_aug)
        assert 0.0 <= prob <= 1.0, f"Probablity must be between 0.0 and 1.0 (given: {prob})"
        self.prob = prob
        self._cnt = 0
        self.clip_frame_cnt = clip_frame_cnt

    def get_transform(self, *args):
        if self._cnt % self.clip_frame_cnt == 0:
            self.do = self._rand_range() < self.prob
            self._cnt = 0   # avoiding overflow
        self._cnt += 1

        if self.do:
            return self.aug.get_transform(*args)
        else:
            return NoOpTransform()

    def __call__(self, aug_input):
        if self._cnt % self.clip_frame_cnt == 0:
            self.do = self._rand_range() < self.prob
            self._cnt = 0   # avoiding overflow
        self._cnt += 1

        if self.do:
            return self.aug(aug_input)
        else:
            return NoOpTransform()


class RandomRotationClip(T.Augmentation):
    """
    This method returns a copy of this image, rotated the given
    number of degrees counter clockwise around the given center.
    """

    def __init__(self, angle, prob=0.5, expand=True, center=None, interp=None, clip_frame_cnt=1):
        """
        Args:
            angle (list[float]): If ``sample_style=="range"``,
                a [min, max] interval from which to sample the angle (in degrees).
                If ``sample_style=="choice"``, a list of angles to sample from
            expand (bool): choose if the image should be resized to fit the whole
                rotated image (default), or simply cropped
            center (list[[float, float]]):  If ``sample_style=="range"``,
                a [[minx, miny], [maxx, maxy]] relative interval from which to sample the center,
                [0, 0] being the top left of the image and [1, 1] the bottom right.
                If ``sample_style=="choice"``, a list of centers to sample from
                Default: None, which means that the center of rotation is the center of the image
                center has no effect if expand=True because it only affects shifting
        """
        super().__init__()
        if isinstance(angle, (float, int)):
            angle = (angle, angle)
        if center is not None and isinstance(center[0], (float, int)):
            center = (center, center)
        self.angle_save = None
        self.center_save = None
        self._cnt = 0
        self._init(locals())

    def get_transform(self, image):
        h, w = image.shape[:2]
        if self._cnt % self.clip_frame_cnt == 0:
            center = None
            angle = np.random.uniform(self.angle[0], self.angle[1], size=self.clip_frame_cnt)
            if self.center is not None:
                center = (
                    np.random.uniform(self.center[0][0], self.center[1][0]),
                    np.random.uniform(self.center[0][1], self.center[1][1]),
                )
            angle = np.sort(angle)
            if self._rand_range() < self.prob:
                angle = angle[::-1]
            self.angle_save = angle
            self.center_save = center

            self._cnt = 0   # avoiding overflow

        angle = self.angle_save[self._cnt]
        center = self.center_save

        self._cnt += 1

        if center is not None:
            center = (w * center[0], h * center[1])  # Convert to absolute coordinates

        if angle % 360 == 0:
            return NoOpTransform()

        return T.RotationTransform(h, w, angle, expand=self.expand, center=center, interp=self.interp)


class ResizeScaleClip(T.Augmentation):
    """
    Takes target size as input and randomly scales the given target size between `min_scale`
    and `max_scale`. It then scales the input image such that it fits inside the scaled target
    box, keeping the aspect ratio constant.
    This implements the resize part of the Google's 'resize_and_crop' data augmentation:
    https://github.com/tensorflow/tpu/blob/master/models/official/detection/utils/input_utils.py#L127
    """

    def __init__(
        self,
        min_scale: float,
        max_scale: float,
        target_height: int,
        target_width: int,
        interp: int = Image.BILINEAR,
        clip_frame_cnt=1,
    ):
        """
        Args:
            min_scale: minimum image scale range.
            max_scale: maximum image scale range.
            target_height: target image height.
            target_width: target image width.
            interp: image interpolation method.
        """
        super().__init__()
        self._init(locals())
        self._cnt = 0

    def _get_resize(self, image: np.ndarray, scale: float):
        input_size = image.shape[:2]

        # Compute new target size given a scale.
        target_size = (self.target_height, self.target_width)
        target_scale_size = np.multiply(target_size, scale)

        # Compute actual rescaling applied to input image and output size.
        output_scale = np.minimum(
            target_scale_size[0] / input_size[0], target_scale_size[1] / input_size[1]
        )
        output_size = np.round(np.multiply(input_size, output_scale)).astype(int)

        return T.ResizeTransform(
            input_size[0], input_size[1], output_size[0], output_size[1], self.interp
        )

    def get_transform(self, image: np.ndarray):
        if self._cnt % self.clip_frame_cnt == 0:
            random_scale = np.random.uniform(self.min_scale, self.max_scale)    
            self.random_scale_save = random_scale

            self._cnt = 0   # avoiding overflow
        self._cnt += 1
        random_scale = self.random_scale_save
        
        return self._get_resize(image, random_scale)


class RandomCropClip(T.Augmentation):
    """
    Randomly crop a rectangle region out of an image.
    """

    def __init__(self, crop_type: str, crop_size, clip_frame_cnt=1):
        """
        Args:
            crop_type (str): one of "relative_range", "relative", "absolute", "absolute_range".
            crop_size (tuple[float, float]): two floats, explained below.
        - "relative": crop a (H * crop_size[0], W * crop_size[1]) region from an input image of
          size (H, W). crop size should be in (0, 1]
        - "relative_range": uniformly sample two values from [crop_size[0], 1]
          and [crop_size[1]], 1], and use them as in "relative" crop type.
        - "absolute" crop a (crop_size[0], crop_size[1]) region from input image.
          crop_size must be smaller than the input image size.
        - "absolute_range", for an input of size (H, W), uniformly sample H_crop in
          [crop_size[0], min(H, crop_size[1])] and W_crop in [crop_size[0], min(W, crop_size[1])].
          Then crop a region (H_crop, W_crop).
        """
        # TODO style of relative_range and absolute_range are not consistent:
        # one takes (h, w) but another takes (min, max)
        super().__init__()
        assert crop_type in ["relative_range", "relative", "absolute", "absolute_range"]
        self._init(locals())
        self._cnt = 0

    def get_transform(self, image):
        h, w = image.shape[:2]      # 667, 500
        if self._cnt % self.clip_frame_cnt == 0:
            croph, cropw = self.get_crop_size((h, w))
            assert h >= croph and w >= cropw, "Shape computation in {} has bugs.".format(self)

            h0 = np.random.randint(h - croph + 1)   # rand(124) -> 5
            w0 = np.random.randint(w - cropw + 1)   # rand(111) -> 634

            h1 = np.random.randint(h0, h - croph + 1)
            w1 = np.random.randint(w0, w - cropw + 1)

            x = np.sort(np.random.rand(self.clip_frame_cnt))

            h = h0 * x + h1 * (1-x)
            w = w0 * x + w1 * (1-x)
            h = np.round_(h).astype(np.int)
            w = np.round_(w).astype(np.int)

            if self._rand_range() < 0.5:
                h = h[::-1]
                w = w[::-1]

            self.hw_save = (h, w)
            self.crop_h_save, self.crop_w_save = croph, cropw
            self._cnt = 0   # avoiding overflow
        _h, _w = self.hw_save[0][self._cnt], self.hw_save[1][self._cnt]
        self._cnt += 1

        return T.CropTransform(_w, _h, self.crop_w_save, self.crop_h_save)

    def get_crop_size(self, image_size):
        """
        Args:
            image_size (tuple): height, width
        Returns:
            crop_size (tuple): height, width in absolute pixels
        """
        h, w = image_size
        if self.crop_type == "relative":
            ch, cw = self.crop_size
            return int(h * ch + 0.5), int(w * cw + 0.5)
        elif self.crop_type == "relative_range":
            crop_size = np.asarray(self.crop_size, dtype=np.float32)
            ch, cw = crop_size + np.random.rand(2) * (1 - crop_size)
            return int(h * ch + 0.5), int(w * cw + 0.5)
        elif self.crop_type == "absolute":
            return (min(self.crop_size[0], h), min(self.crop_size[1], w))
        elif self.crop_type == "absolute_range":
            assert self.crop_size[0] <= self.crop_size[1]
            ch = np.random.randint(min(h, self.crop_size[0]), min(h, self.crop_size[1]) + 1)
            cw = np.random.randint(min(w, self.crop_size[0]), min(w, self.crop_size[1]) + 1)
            return ch, cw
        else:
            raise NotImplementedError("Unknown crop type {}".format(self.crop_type))


class FixedSizeCropClip(T.Augmentation):
    """
    If `crop_size` is smaller than the input image size, then it uses a random crop of
    the crop size. If `crop_size` is larger than the input image size, then it pads
    the right and the bottom of the image to the crop size if `pad` is True, otherwise
    it returns the smaller image.
    """

    def __init__(self, crop_size: Tuple[int], pad: bool = True, pad_value: float = 128.0, clip_frame_cnt=1):
        """
        Args:
            crop_size: target image (height, width).
            pad: if True, will pad images smaller than `crop_size` up to `crop_size`
            pad_value: the padding value.
        """
        super().__init__()
        self._init(locals())
        self._cnt = 0

    def _get_crop(self, image: np.ndarray):
        # Compute the image scale and scaled size.
        input_size = image.shape[:2]
        output_size = self.crop_size

        # Add random crop if the image is scaled up.
        max_offset = np.subtract(input_size, output_size)
        max_offset = np.maximum(max_offset, 0)

        if self._cnt % self.clip_frame_cnt == 0:
            offset = np.multiply(max_offset, np.random.uniform(0.0, 1.0))
            offset = np.round(offset).astype(int)
            self.offset_save = offset
            self._cnt = 0   # avoiding overflow
        self._cnt += 1
        offset = self.offset_save
        return CropTransform(
            offset[1], offset[0], output_size[1], output_size[0], input_size[1], input_size[0]
        )

    def _get_pad(self, image: np.ndarray):
        # Compute the image scale and scaled size.
        input_size = image.shape[:2]
        output_size = self.crop_size

        # Add padding if the image is scaled down.
        pad_size = np.subtract(output_size, input_size)
        pad_size = np.maximum(pad_size, 0)
        original_size = np.minimum(input_size, output_size)
        return PadTransform(
            0, 0, pad_size[1], pad_size[0], original_size[1], original_size[0], self.pad_value
        )

    def get_transform(self, image: np.ndarray):
        transforms = [self._get_crop(image)]
        if self.pad:
            transforms.append(self._get_pad(image))
        return TransformList(transforms)


class ResizeShortestEdgeClip(T.Augmentation):
    """
    Scale the shorter edge to the given size, with a limit of `max_size` on the longer edge.
    If `max_size` is reached, then downscale so that the longer edge does not exceed max_size.
    """

    def __init__(
        self, short_edge_length, max_size=sys.maxsize, sample_style="range", interp=Image.BILINEAR, clip_frame_cnt=1
    ):
        """
        Args:
            short_edge_length (list[int]): If ``sample_style=="range"``,
                a [min, max] interval from which to sample the shortest edge length.
                If ``sample_style=="choice"``, a list of shortest edge lengths to sample from.
            max_size (int): maximum allowed longest edge length.
            sample_style (str): either "range" or "choice".
        """
        super().__init__()
        assert sample_style in ["range", "choice", "range_by_clip", "choice_by_clip"], sample_style

        self.is_range = ("range" in sample_style)
        if isinstance(short_edge_length, int):
            short_edge_length = (short_edge_length, short_edge_length)
        if self.is_range:
            assert len(short_edge_length) == 2, (
                "short_edge_length must be two values using 'range' sample style."
                f" Got {short_edge_length}!"
            )
        self._cnt = 0
        self._init(locals())

    def get_transform(self, image):
        if self._cnt % self.clip_frame_cnt == 0:
            if self.is_range:
                self.size = np.random.randint(self.short_edge_length[0], self.short_edge_length[1] + 1)
            else:
                self.size = np.random.choice(self.short_edge_length)
            self._cnt = 0   # avoiding overflow

            if self.size == 0:
                return NoOpTransform()
        self._cnt += 1

        h, w = image.shape[:2]

        scale = self.size * 1.0 / min(h, w)
        if h < w:
            newh, neww = self.size, scale * w
        else:
            newh, neww = scale * h, self.size
        if max(newh, neww) > self.max_size:
            scale = self.max_size * 1.0 / max(newh, neww)
            newh = newh * scale
            neww = neww * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return T.ResizeTransform(h, w, newh, neww, self.interp)


class RandomFlipClip(T.Augmentation):
    """
    Flip the image horizontally or vertically with the given probability.
    """

    def __init__(self, prob=0.5, *, horizontal=True, vertical=False, clip_frame_cnt=1):
        """
        Args:
            prob (float): probability of flip.
            horizontal (boolean): whether to apply horizontal flipping
            vertical (boolean): whether to apply vertical flipping
        """
        super().__init__()

        if horizontal and vertical:
            raise ValueError("Cannot do both horiz and vert. Please use two Flip instead.")
        if not horizontal and not vertical:
            raise ValueError("At least one of horiz or vert has to be True!")
        self._cnt = 0

        self._init(locals())

    def get_transform(self, image):
        if self._cnt % self.clip_frame_cnt == 0:
            self.do = self._rand_range() < self.prob
            self._cnt = 0   # avoiding overflow
        self._cnt += 1

        h, w = image.shape[:2]

        if self.do:
            if self.horizontal:
                return HFlipTransform(w)
            elif self.vertical:
                return VFlipTransform(h)
        else:
            return NoOpTransform()


def build_augmentation(cfg, is_train):
    logger = logging.getLogger(__name__)
    aug_list = []
    if is_train:
        use_lsj = cfg.INPUT.LSJ_AUG.ENABLED
        if use_lsj:
            image_size = cfg.INPUT.LSJ_AUG.IMAGE_SIZE
            min_scale = cfg.INPUT.LSJ_AUG.MIN_SCALE
            max_scale = cfg.INPUT.LSJ_AUG.MAX_SCALE

            if cfg.INPUT.RANDOM_FLIP != "none":
                if cfg.INPUT.RANDOM_FLIP == "flip_by_clip":
                    flip_clip_frame_cnt = cfg.INPUT.SAMPLING_FRAME_NUM
                else:
                    flip_clip_frame_cnt = 1

                aug_list.append(
                    # NOTE using RandomFlip modified for the support of flip maintenance
                    RandomFlipClip(
                        horizontal=(cfg.INPUT.RANDOM_FLIP == "horizontal") or (cfg.INPUT.RANDOM_FLIP == "flip_by_clip"),
                        vertical=cfg.INPUT.RANDOM_FLIP == "vertical",
                        clip_frame_cnt=flip_clip_frame_cnt,
                    )
                )

            aug_list.extend([
                T.ResizeScale(
                    min_scale=min_scale, max_scale=max_scale, target_height=image_size, target_width=image_size
                ),
                T.FixedSizeCrop(crop_size=(image_size, image_size)),
            ])

        else:
            min_size = cfg.INPUT.MIN_SIZE_TRAIN
            max_size = cfg.INPUT.MAX_SIZE_TRAIN
            sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
            clip_frame_cnt = cfg.INPUT.SAMPLING_FRAME_NUM if "by_clip" in cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING else 1

            # Crop
            if cfg.INPUT.CROP.ENABLED:
                crop_aug = RandomApplyClip(
                    T.AugmentationList([
                        ResizeShortestEdgeClip([400, 500, 600], 1333, sample_style, clip_frame_cnt=clip_frame_cnt),
                        RandomCropClip(cfg.INPUT.PSEUDO.CROP.TYPE, cfg.INPUT.PSEUDO.CROP.SIZE, clip_frame_cnt=clip_frame_cnt)
                    ]),
                    clip_frame_cnt=clip_frame_cnt
                )
                aug_list.append(crop_aug)

            # Resize
            aug_list.append(ResizeShortestEdgeClip(min_size, max_size, sample_style, clip_frame_cnt=clip_frame_cnt))

            # Flip
            if cfg.INPUT.RANDOM_FLIP != "none":
                if cfg.INPUT.RANDOM_FLIP == "flip_by_clip":
                    flip_clip_frame_cnt = cfg.INPUT.SAMPLING_FRAME_NUM
                else:
                    flip_clip_frame_cnt = 1

                aug_list.append(
                    # NOTE using RandomFlip modified for the support of flip maintenance
                    RandomFlipClip(
                        horizontal=(cfg.INPUT.RANDOM_FLIP == "horizontal") or (cfg.INPUT.RANDOM_FLIP == "flip_by_clip"),
                        vertical=cfg.INPUT.RANDOM_FLIP == "vertical",
                        clip_frame_cnt=flip_clip_frame_cnt,
                    )
                )

            # Additional augmentations : brightness, contrast, saturation, rotation
            augmentations = cfg.INPUT.AUGMENTATIONS
            if "brightness" in augmentations:
                aug_list.append(T.RandomBrightness(0.9, 1.1))
            if "contrast" in augmentations:
                aug_list.append(T.RandomContrast(0.9, 1.1))
            if "saturation" in augmentations:
                aug_list.append(T.RandomSaturation(0.9, 1.1))
            if "rotation" in augmentations:
                aug_list.append(
                    T.RandomRotation(
                        [-15, 15], expand=False, center=[(0.4, 0.4), (0.6, 0.6)], sample_style="range"
                    )
                )
    else:
        # Resize
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        sample_style = "choice"
        aug_list.append(T.ResizeShortestEdge(min_size, max_size, sample_style))

    return aug_list


def build_pseudo_augmentation(cfg, is_train):
    logger = logging.getLogger(__name__)
    aug_list = []
    if is_train:
        use_lsj = cfg.INPUT.LSJ_AUG.ENABLED
        if use_lsj:
            image_size = cfg.INPUT.LSJ_AUG.IMAGE_SIZE
            min_scale = cfg.INPUT.LSJ_AUG.MIN_SCALE
            max_scale = cfg.INPUT.LSJ_AUG.MAX_SCALE

            if cfg.INPUT.RANDOM_FLIP != "none":
                if cfg.INPUT.RANDOM_FLIP == "flip_by_clip":
                    clip_frame_cnt = cfg.INPUT.SAMPLING_FRAME_NUM
                else:
                    clip_frame_cnt = 1

                aug_list.append(
                    # NOTE using RandomFlip modified for the support of flip maintenance
                    RandomFlipClip(
                        horizontal=(cfg.INPUT.RANDOM_FLIP == "horizontal") or (cfg.INPUT.RANDOM_FLIP == "flip_by_clip"),
                        vertical=cfg.INPUT.RANDOM_FLIP == "vertical",
                        clip_frame_cnt=clip_frame_cnt,
                    )
                )

            # Additional augmentations : brightness, contrast, saturation, rotation
            augmentations = cfg.INPUT.PSEUDO.AUGMENTATIONS
            if "brightness" in augmentations:
                aug_list.append(T.RandomBrightness(0.9, 1.1))
            if "contrast" in augmentations:
                aug_list.append(T.RandomContrast(0.9, 1.1))
            if "saturation" in augmentations:
                aug_list.append(T.RandomSaturation(0.9, 1.1))
            if "rotation" in augmentations:
                aug_list.append(
                    RandomRotationClip(
                        [-15, 15], expand=False, center=[(0.4, 0.4), (0.6, 0.6)], clip_frame_cnt=clip_frame_cnt,
                    )
                )

            aug_list.extend([
                ResizeScaleClip(
                    min_scale=min_scale, max_scale=max_scale, target_height=image_size, target_width=image_size,
                    clip_frame_cnt=clip_frame_cnt,
                ),
                FixedSizeCropClip(crop_size=(image_size, image_size), clip_frame_cnt=clip_frame_cnt),
            ])
        else:
            min_size = cfg.INPUT.PSEUDO.MIN_SIZE_TRAIN
            max_size = cfg.INPUT.PSEUDO.MAX_SIZE_TRAIN
            sample_style = cfg.INPUT.PSEUDO.MIN_SIZE_TRAIN_SAMPLING
            clip_frame_cnt = cfg.INPUT.SAMPLING_FRAME_NUM

            # Crop
            if cfg.INPUT.PSEUDO.CROP.ENABLED:
                crop_aug = RandomApplyClip(
                    T.AugmentationList([
                        ResizeShortestEdgeClip([400, 500, 600], 1333, sample_style, clip_frame_cnt=clip_frame_cnt),
                        RandomCropClip(cfg.INPUT.PSEUDO.CROP.TYPE, cfg.INPUT.PSEUDO.CROP.SIZE, clip_frame_cnt=clip_frame_cnt)
                    ]),
                    clip_frame_cnt=clip_frame_cnt
                )
                aug_list.append(crop_aug)

            # Resize
            aug_list.append(ResizeShortestEdgeClip(min_size, max_size, sample_style, clip_frame_cnt=clip_frame_cnt))

            # Flip
            aug_list.append(
                # NOTE using RandomFlip modified for the support of flip maintenance
                RandomFlipClip(
                    horizontal=(cfg.INPUT.RANDOM_FLIP == "horizontal") or (cfg.INPUT.RANDOM_FLIP == "flip_by_clip"),
                    vertical=cfg.INPUT.RANDOM_FLIP == "vertical",
                    clip_frame_cnt=clip_frame_cnt,
                )
            )

            # Additional augmentations : brightness, contrast, saturation, rotation
            augmentations = cfg.INPUT.PSEUDO.AUGMENTATIONS
            if "brightness" in augmentations:
                aug_list.append(T.RandomBrightness(0.9, 1.1))
            if "contrast" in augmentations:
                aug_list.append(T.RandomContrast(0.9, 1.1))
            if "saturation" in augmentations:
                aug_list.append(T.RandomSaturation(0.9, 1.1))
            if "rotation" in augmentations:
                aug_list.append(
                    RandomRotationClip(
                        [-15, 15], expand=False, center=[(0.4, 0.4), (0.6, 0.6)], clip_frame_cnt=clip_frame_cnt,
                    )
                )
    else:
        # Resize
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        sample_style = "choice"
        aug_list.append(T.ResizeShortestEdge(min_size, max_size, sample_style))

    return aug_list
