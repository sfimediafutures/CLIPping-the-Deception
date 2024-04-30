"""
Modified from https://github.com/KaiyangZhou/deep-person-reid
"""
import os
import sys
import json
import time
import errno
import numpy as np
import random
import os.path as osp
import warnings
from difflib import SequenceMatcher
import PIL
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter
import cv2
from random import choice, shuffle
from io import BytesIO

__all__ = [
    "mkdir_if_missing",
    "check_isfile",
    "read_json",
    "write_json",
    "set_random_seed",
    "download_url",
    "read_image",
    "collect_env_info",
    "listdir_nohidden",
    "get_most_similar_str_to_a_from_b",
    "check_availability",
    "tolist_if_not",
]


def mkdir_if_missing(dirname):
    """Create dirname if it is missing."""
    if not osp.exists(dirname):
        try:
            os.makedirs(dirname)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def check_isfile(fpath):
    """Check if the given path is a file.

    Args:
        fpath (str): file path.

    Returns:
       bool
    """
    isfile = osp.isfile(fpath)
    if not isfile:
        warnings.warn('No file found at "{}"'.format(fpath))
    return isfile


def read_json(fpath):
    """Read json file from a path."""
    with open(fpath, "r") as f:
        obj = json.load(f)
    return obj


def write_json(obj, fpath):
    """Writes to a json file."""
    mkdir_if_missing(osp.dirname(fpath))
    with open(fpath, "w") as f:
        json.dump(obj, f, indent=4, separators=(",", ": "))


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def download_url(url, dst):
    """Download file from a url to a destination.

    Args:
        url (str): url to download file.
        dst (str): destination path.
    """
    from six.moves import urllib

    print('* url="{}"'.format(url))
    print('* destination="{}"'.format(dst))

    def _reporthook(count, block_size, total_size):
        global start_time
        if count == 0:
            start_time = time.time()
            return
        duration = time.time() - start_time
        progress_size = int(count * block_size)
        speed = int(progress_size / (1024*duration))
        percent = int(count * block_size * 100 / total_size)
        sys.stdout.write(
            "\r...%d%%, %d MB, %d KB/s, %d seconds passed" %
            (percent, progress_size / (1024*1024), speed, duration)
        )
        sys.stdout.flush()

    urllib.request.urlretrieve(url, dst, _reporthook)
    sys.stdout.write("\n")


def sample_continuous(s):
    if len(s) == 1:
        return s[0]
    if len(s) == 2:
        rg = s[1] - s[0]
        return random.random() * rg + s[0]
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
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    result, encimg = cv2.imencode('.jpg', img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg[:,:,::-1]

def pil_jpg(img, compress_val):
    out = BytesIO()
    img = Image.fromarray(img)
    img.save(out, format='jpeg', quality=compress_val)
    img = Image.open(out)
    # load from memory before ByteIO closes
    img = np.array(img)
    out.close()
    return img

jpeg_dict = {'cv2': cv2_jpg, 'pil': pil_jpg}
def jpeg_from_key(img, compress_val, key):
    method = jpeg_dict[key]
    return method(img, compress_val)

def data_augment_train(img):
    blur_prob = 0.1
    jpg_prob = 0.4
    blur_sig = [0.0,2.0]
    jpg_method = ['cv2','pil']
    jpg_qual = [30,100]
    
    img = np.array(img)
    # if img.ndim == 2:
    #     img = np.expand_dims(img, axis=2)
    #     img = np.repeat(img, 3, axis=2)
    
    if random.random() < blur_prob:
        sig = sample_continuous(blur_sig)
        gaussian_blur(img, sig)

    if random.random() < jpg_prob:
        method = sample_discrete(jpg_method)
        qual = sample_discrete(jpg_qual)
        img = jpeg_from_key(img, qual, method)
        # print(img.shape)
    # print(img.shape)
    return Image.fromarray(img)

def data_augment_test(img):
    blur_prob = 1.0
    jpg_prob = 0.0
    blur_sig = [2.0]
    jpg_method = ['cv2','pil']
    jpg_qual = [75]
    
    img = np.array(img)
    # if img.ndim == 2:
    #     img = np.expand_dims(img, axis=2)
    #     img = np.repeat(img, 3, axis=2)
    
    if random.random() < blur_prob:
        sig = sample_continuous(blur_sig)
        gaussian_blur(img, sig)

    if random.random() <= jpg_prob:
        method = sample_discrete(jpg_method)
        qual = sample_discrete(jpg_qual)
        img = jpeg_from_key(img, qual, method)
        # print(img.shape)
    # print(img.shape)
    return Image.fromarray(img)


def read_image(path):
    """Read image from path using ``PIL.Image``.

    Args:
        path (str): path to an image.

    Returns:
        PIL image
    """
    # img = Image.open(path).convert("RGB")
    # return data_augment_test(img)
    
    return Image.open(path).convert("RGB")


def collect_env_info():
    """Return env info as a string.
    Code source: github.com/facebookresearch/maskrcnn-benchmark
    """
    from torch.utils.collect_env import get_pretty_env_info

    env_str = get_pretty_env_info()
    env_str += "\n        Pillow ({})".format(PIL.__version__)
    return env_str


def listdir_nohidden(path, sort=False):
    """List non-hidden items in a directory.

    Args:
         path (str): directory path.
         sort (bool): sort the items.
    """
    items = [f for f in os.listdir(path) if not f.startswith(".")]
    if sort:
        items.sort()
    return items


def get_most_similar_str_to_a_from_b(a, b):
    """Return the most similar string to a in b.

    Args:
        a (str): probe string.
        b (list): a list of candidate strings.
    """
    highest_sim = 0
    chosen = None
    for candidate in b:
        sim = SequenceMatcher(None, a, candidate).ratio()
        if sim >= highest_sim:
            highest_sim = sim
            chosen = candidate
    return chosen


def check_availability(requested, available):
    """Check if an element is available in a list.

    Args:
        requested (str): probe string.
        available (list): a list of available strings.
    """
    if requested not in available:
        psb_ans = get_most_similar_str_to_a_from_b(requested, available)
        raise ValueError(
            "The requested one is expected "
            "to belong to {}, but got [{}] "
            "(do you mean [{}]?)".format(available, requested, psb_ans)
        )


def tolist_if_not(x):
    """Convert to a list."""
    if not isinstance(x, list):
        x = [x]
    return x
