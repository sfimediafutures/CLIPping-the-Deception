import numpy as np
import random
import torch
import torchvision.transforms.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter
import cv2
from io import BytesIO
from random import random, choice, shuffle
from torchvision.transforms import (
    Resize, Compose, ToTensor, Normalize, CenterCrop, RandomCrop, ColorJitter,
    RandomApply, GaussianBlur, RandomGrayscale, RandomResizedCrop,
    RandomHorizontalFlip
)
from torchvision.transforms.functional import InterpolationMode

from .autoaugment import SVHNPolicy, CIFAR10Policy, ImageNetPolicy
from .randaugment import RandAugment, RandAugment2, RandAugmentFixMatch

AVAI_CHOICES = [
    "random_flip",
    "random_resized_crop",
    "normalize",
    "instance_norm",
    "random_crop",
    "random_translation",
    "center_crop",  # This has become a default operation during testing
    "cutout",
    "imagenet_policy",
    "cifar10_policy",
    "svhn_policy",
    "randaugment",
    "randaugment_fixmatch",
    "randaugment2",
    "gaussian_noise",
    "colorjitter",
    "randomgrayscale",
    "gaussian_blur",
    "compress_blur",
]

INTERPOLATION_MODES = {
    "bilinear": InterpolationMode.BILINEAR,
    "bicubic": InterpolationMode.BICUBIC,
    "nearest": InterpolationMode.NEAREST,
}


def sample_continuous(s):
    if len(s) == 1:
        return s[0]
    if len(s) == 2:
        rg = s[1] - s[0]
        return random() * rg + s[0]
    raise ValueError("Length of iterable s should be 1 or 2.")


def sample_discrete(s):
    if len(s) == 1:
        return s[0]
    return choice(s)


def gaussian_blur(img, sigma):
    gaussian_filter(img[:,:,0], output=img[:,:,0], sigma=sigma)
    gaussian_filter(img[:,:,1], output=img[:,:,1], sigma=sigma)
    gaussian_filter(img[:,:,2], output=img[:,:,2], sigma=sigma)


def cv2_jpg(img, compress_val):
    img_cv2 = img[:,:,::-1]
    img_cv2 = np.transpose(img_cv2, (1, 2, 0))
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    result, encimg = cv2.imencode('.jpg', img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg[:,:,::-1]


# def pil_jpg(img, compress_val):
#     print('IN PIL')
#     out = BytesIO()
#     img = Image.fromarray(img)
#     # img = np.transpose(img, (1, 2, 0))
#     img.save(out, format='jpeg', quality=compress_val)
#     img = Image.open(out)
#     # load from memory before ByteIO closes
#     img = np.array(img)
#     print(img.shape)
#     out.close()
#     return img


# jpeg_dict = {'cv2': cv2_jpg, 'pil': pil_jpg}
# def jpeg_from_key(img, compress_val, key):
#     method = jpeg_dict[key]
#     return method(img, compress_val)

class Random2DTranslation:
    """Given an image of (height, width), we resize it to
    (height*1.125, width*1.125), and then perform random cropping.

    Args:
        height (int): target image height.
        width (int): target image width.
        p (float, optional): probability that this operation takes place.
            Default is 0.5.
        interpolation (int, optional): desired interpolation. Default is
            ``torchvision.transforms.functional.InterpolationMode.BILINEAR``
    """

    def __init__(
        self, height, width, p=0.5, interpolation=InterpolationMode.BILINEAR
    ):
        self.height = height
        self.width = width
        self.p = p
        self.interpolation = interpolation

    def __call__(self, img):
        if random.uniform(0, 1) > self.p:
            return F.resize(
                img=img,
                size=[self.height, self.width],
                interpolation=self.interpolation
            )

        new_width = int(round(self.width * 1.125))
        new_height = int(round(self.height * 1.125))
        resized_img = F.resize(
            img=img,
            size=[new_height, new_width],
            interpolation=self.interpolation
        )
        x_maxrange = new_width - self.width
        y_maxrange = new_height - self.height
        x1 = int(round(random.uniform(0, x_maxrange)))
        y1 = int(round(random.uniform(0, y_maxrange)))
        croped_img = F.crop(
            img=resized_img,
            top=y1,
            left=x1,
            height=self.height,
            width=self.width
        )

        return croped_img


class InstanceNormalization:
    """Normalize data using per-channel mean and standard deviation.

    Reference:
        - Ulyanov et al. Instance normalization: The missing in- gredient
          for fast stylization. ArXiv 2016.
        - Shu et al. A DIRT-T Approach to Unsupervised Domain Adaptation.
          ICLR 2018.
    """

    def __init__(self, eps=1e-8):
        self.eps = eps

    def __call__(self, img):
        C, H, W = img.shape
        img_re = img.reshape(C, H * W)
        mean = img_re.mean(1).view(C, 1, 1)
        std = img_re.std(1).view(C, 1, 1)
        return (img-mean) / (std + self.eps)


class Cutout:
    """Randomly mask out one or more patches from an image.

    https://github.com/uoguelph-mlrg/Cutout

    Args:
        n_holes (int, optional): number of patches to cut out
            of each image. Default is 1.
        length (int, optinal): length (in pixels) of each square
            patch. Default is 16.
    """

    def __init__(self, n_holes=1, length=16):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): tensor image of size (C, H, W).

        Returns:
            Tensor: image with n_holes of dimension
                length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1:y2, x1:x2] = 0.0

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        return img * mask


class GaussianNoise:
    """Add gaussian noise."""

    def __init__(self, mean=0, std=0.15, p=0.5):
        self.mean = mean
        self.std = std
        self.p = p

    def __call__(self, img):
        if random.uniform(0, 1) > self.p:
            return img
        noise = torch.randn(img.size()) * self.std + self.mean
        return img + noise
    

class CompressionBlurring:
    """Compress and blur images randomly."""

    def __init__(self):
        self.blur_prob = 0.5
        self.jpg_prob = 0.5
        self.blur_sig = [0.0,3.0]
        self.jpg_method = ['cv2']
        self.jpg_qual = [30,100]
        # self.p = p

    def __call__(self, img):
        # print(img.shape)
        img = np.array(img)
        if random() < self.blur_prob:
            sig = sample_continuous(self.blur_sig)
            gaussian_blur(img, sig)

        if random() < self.jpg_prob:
            method = sample_discrete(self.jpg_method)
            qual = sample_discrete(self.jpg_qual)
            # img = jpeg_from_key(img, qual, method)
            img = cv2_jpg(img, qual)
        return Image.fromarray(img)


def build_transform(cfg, is_train=True, choices=None):
    """Build transformation function.

    Args:
        cfg (CfgNode): config.
        is_train (bool, optional): for training (True) or test (False).
            Default is True.
        choices (list, optional): list of strings which will overwrite
            cfg.INPUT.TRANSFORMS if given. Default is None.
    """
    if cfg.INPUT.NO_TRANSFORM:
        print("Note: no transform is applied!")
        return None

    if choices is None:
        choices = cfg.INPUT.TRANSFORMS

    for choice in choices:
        assert choice in AVAI_CHOICES

    target_size = f"{cfg.INPUT.SIZE[0]}x{cfg.INPUT.SIZE[1]}"

    normalize = Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)

    if is_train:
        return _build_transform_train(cfg, choices, target_size, normalize)
    else:
        return _build_transform_test(cfg, choices, target_size, normalize)


def _build_transform_train(cfg, choices, target_size, normalize):
    print("Building transform_train")
    tfm_train = []

    interp_mode = INTERPOLATION_MODES[cfg.INPUT.INTERPOLATION]
    input_size = cfg.INPUT.SIZE

    # Make sure the image size matches the target size
    conditions = []
    conditions += ["random_crop" not in choices]
    conditions += ["random_resized_crop" not in choices]
    if all(conditions):
        print(f"+ resize to {target_size}")
        tfm_train += [Resize(input_size, interpolation=interp_mode)]

    if "random_translation" in choices:
        print("+ random translation")
        tfm_train += [Random2DTranslation(input_size[0], input_size[1])]

    if "random_crop" in choices:
        crop_padding = cfg.INPUT.CROP_PADDING
        print(f"+ random crop (padding = {crop_padding})")
        tfm_train += [RandomCrop(input_size, padding=crop_padding)]

    if "random_resized_crop" in choices:
        s_ = cfg.INPUT.RRCROP_SCALE
        print(f"+ random resized crop (size={input_size}, scale={s_})")
        tfm_train += [
            RandomResizedCrop(input_size, scale=s_, interpolation=interp_mode)
        ]

    if "random_flip" in choices:
        print("+ random flip")
        tfm_train += [RandomHorizontalFlip()]

    if "imagenet_policy" in choices:
        print("+ imagenet policy")
        tfm_train += [ImageNetPolicy()]

    if "cifar10_policy" in choices:
        print("+ cifar10 policy")
        tfm_train += [CIFAR10Policy()]

    if "svhn_policy" in choices:
        print("+ svhn policy")
        tfm_train += [SVHNPolicy()]

    if "randaugment" in choices:
        n_ = cfg.INPUT.RANDAUGMENT_N
        m_ = cfg.INPUT.RANDAUGMENT_M
        print(f"+ randaugment (n={n_}, m={m_})")
        tfm_train += [RandAugment(n_, m_)]

    if "randaugment_fixmatch" in choices:
        n_ = cfg.INPUT.RANDAUGMENT_N
        print(f"+ randaugment_fixmatch (n={n_})")
        tfm_train += [RandAugmentFixMatch(n_)]

    if "randaugment2" in choices:
        n_ = cfg.INPUT.RANDAUGMENT_N
        print(f"+ randaugment2 (n={n_})")
        tfm_train += [RandAugment2(n_)]

    if "colorjitter" in choices:
        b_ = cfg.INPUT.COLORJITTER_B
        c_ = cfg.INPUT.COLORJITTER_C
        s_ = cfg.INPUT.COLORJITTER_S
        h_ = cfg.INPUT.COLORJITTER_H
        print(
            f"+ color jitter (brightness={b_}, "
            f"contrast={c_}, saturation={s_}, hue={h_})"
        )
        tfm_train += [
            ColorJitter(
                brightness=b_,
                contrast=c_,
                saturation=s_,
                hue=h_,
            )
        ]

    if "randomgrayscale" in choices:
        print("+ random gray scale")
        tfm_train += [RandomGrayscale(p=cfg.INPUT.RGS_P)]

    if "gaussian_blur" in choices:
        print(f"+ gaussian blur (kernel={cfg.INPUT.GB_K})")
        gb_k, gb_p = cfg.INPUT.GB_K, cfg.INPUT.GB_P
        tfm_train += [RandomApply([GaussianBlur(gb_k)], p=gb_p)]

    print("+ to torch tensor of range [0, 1]")
    tfm_train += [ToTensor()]

    if "cutout" in choices:
        cutout_n = cfg.INPUT.CUTOUT_N
        cutout_len = cfg.INPUT.CUTOUT_LEN
        print(f"+ cutout (n_holes={cutout_n}, length={cutout_len})")
        tfm_train += [Cutout(cutout_n, cutout_len)]

    if "normalize" in choices:
        print(
            f"+ normalization (mean={cfg.INPUT.PIXEL_MEAN}, std={cfg.INPUT.PIXEL_STD})"
        )
        tfm_train += [normalize]

    if "gaussian_noise" in choices:
        print(
            f"+ gaussian noise (mean={cfg.INPUT.GN_MEAN}, std={cfg.INPUT.GN_STD})"
        )
        tfm_train += [GaussianNoise(cfg.INPUT.GN_MEAN, cfg.INPUT.GN_STD)]

    if "instance_norm" in choices:
        print("+ instance normalization")
        tfm_train += [InstanceNormalization()]

    # CompressionBlurring
    # if "compress_blur":
    #     print("Applying compression and blurring")
    #     tfm_train += [CompressionBlurring()]

    tfm_train = Compose(tfm_train)

    return tfm_train


def _build_transform_test(cfg, choices, target_size, normalize):
    print("Building transform_test")
    tfm_test = []

    interp_mode = INTERPOLATION_MODES[cfg.INPUT.INTERPOLATION]
    input_size = cfg.INPUT.SIZE

    print(f"+ resize the smaller edge to {max(input_size)}")
    tfm_test += [Resize(max(input_size), interpolation=interp_mode)]

    print(f"+ {target_size} center crop")
    tfm_test += [CenterCrop(input_size)]

    print("+ to torch tensor of range [0, 1]")
    tfm_test += [ToTensor()]

    if "normalize" in choices:
        print(
            f"+ normalization (mean={cfg.INPUT.PIXEL_MEAN}, std={cfg.INPUT.PIXEL_STD})"
        )
        tfm_test += [normalize]

    if "instance_norm" in choices:
        print("+ instance normalization")
        tfm_test += [InstanceNormalization()]

    tfm_test = Compose(tfm_test)

    return tfm_test
