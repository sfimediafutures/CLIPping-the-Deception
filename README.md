# CLIPping the Deception: Adapting Vision Language Models for Universal Deepfake Detection
Code and pre-trained models for our paper, [CLIPping the Deception: Adapting Vision-Language Models for Universal Deepfake Detection](https://arxiv.org/pdf/2402.12927).

## TODO
* Inference code.
* Training code. 

## Installation Guide
This code is built on top of [Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch), so you need to install the `dassl` environment first. Simply follow the instructions described here to install `dassl` as well as PyTorch. 

After installing dassl, you also need to install `CoOp` by following instructions [here](https://github.com/KaiyangZhou/CoOp/tree/main). Run `pip install -r requirements.txt` under `CoOp/` to install a few more packages required by CLIP (this should be done when dassl is activated). Then, you are ready to go.

Follow DATASETS.md to install the datasets.

## Citations
If you use this code in your research, please kindly cite the following papers:
```
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
