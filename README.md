# CLIPping the Deception: Adapting Vision Language Models for Universal Deepfake Detection
Code and pre-trained models for our paper, [CLIPping the Deception: Adapting Vision-Language Models for Universal Deepfake Detection](https://arxiv.org/pdf/2402.12927).

## TODO
* Inference code.
* Evaluation code.
* Training code.

## Evaluation Dataset
The evaluation dataset can be found on: https://tinyurl.com/5b3fh7fh

The dataset is processed as required by [Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch). For each subset, e.g., StarGAN, two folders can be found each containing **real** - **fake** images. The sample path for StarGAN's evaluation images would be something like: `stargan/images/val/`. In the **val** folder there are two folders, (1) `n01440764` - containing **real** images, (2) `n01443537` - containing **fake** images.

## Pre-trained Models
Model weights can be found on: https://tinyurl.com/5dsmpnm7

Download and extract **weights.zip** in the same folder as `evaluate.py`

More details about this convention will be included with the **Inference** and **Evaluation** codes (which will be uploaded soon).

## Installation Guide
This code is built on top of [Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch), so you need to install the `dassl` environment first. Simply follow the instructions described here to install `dassl` as well as PyTorch. 

After installing dassl, you also need to install `CoOp` by following instructions [here](https://github.com/KaiyangZhou/CoOp/tree/main). Run `pip install -r requirements.txt` under `CoOp/` to install a few more packages required by CLIP (this should be done when dassl is activated). Then, you are ready to go.

Follow `DATASETS.md` to install the datasets.

## Citations
If you use this code in your research, please kindly cite the following papers:
```
@article{khan2024clipping,
  title={CLIPping the Deception: Adapting Vision-Language Models for Universal Deepfake Detection},
  author={Khan, Sohail Ahmed and Dang-Nguyen, Duc-Tien},
  journal={arXiv preprint arXiv:2402.12927},
  year={2024}
}

@inproceedings{zhou2022cocoop,
    title={Conditional Prompt Learning for Vision-Language Models},
    author={Zhou, Kaiyang and Yang, Jingkang and Loy, Chen Change and Liu, Ziwei},
    booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2022}
}

@article{zhou2022coop,
    title={Learning to Prompt for Vision-Language Models},
    author={Zhou, Kaiyang and Yang, Jingkang and Loy, Chen Change and Liu, Ziwei},
    journal={International Journal of Computer Vision (IJCV)},
    year={2022}
}
```
