# HisynSeg

HisynSeg: Weakly-Supervised Histopathological Image Segmentation via Image-Mixing Synthesis and Consistency Regularization

Accepted by **IEEE Transactions on Medical Imaging**.

It is an extended version of our AAAI paper **"Weakly-Supervised Semantic Segmentation for Histopathology Images Based on Dataset Synthesis and Feature Consistency Constraint"**.

## Abstract
Tissue semantic segmentation is one of the key tasks in computational pathology. To avoid the expensive and laborious acquisition of pixel-level annotations, a wide range of studies attempt to adopt the class activation map (CAM), a weakly-supervised learning scheme, to achieve pixel-level tissue segmentation. However, CAM-based methods are prone to suffer from under-activation and over-activation issues, leading to poor segmentation performance. To address this problem, we propose a novel weakly-supervised semantic segmentation framework for histopathological images based on image-mixing synthesis and consistency regularization, dubbed HisynSeg. Specifically, synthesized histopathological images with pixel-level masks are generated for fully-supervised model training, where two synthesis strategies are proposed based on Mosaic transformation and Bézier mask generation. Besides, an image filtering module is developed to guarantee the authenticity of the synthesized images. In order to further avoid the model overfitting to the occasional synthesis artifacts, we additionally propose a novel self-supervised consistency regularization, which enables the real images without segmentation masks to supervise the training of the segmentation model. By integrating the proposed techniques, the HisynSeg framework successfully transforms the weakly-supervised semantic segmentation problem into a fully-supervised one, greatly improving the segmentation accuracy. Experimental results on three datasets prove that the proposed method achieves a state-of-the-art performance. Code is available at https://github.com/Vison307/HisynSeg.

## Environment

Code tested on
* Ubuntu 18.04
* A single Nvidia GeForce RTX 3090
* Python 3.8
* Pytorch 1.12.1
* Pytorch Lightning 1.7.1
* Albumentations 1.2.1
* Segmentation models pytorch 0.3.3
* Timm 0.9.2

Please use the follwing command to install the dependencies:

`conda env create -f environment.yaml`

For more details, you can check `Dockerfile` and `requirements.in` for reference.

## Orginal Dataset Preparation

1. Download the [WSSS4LUAD dataset](https://wsss4luad.grand-challenge.org/) and put it in ./data/WSSS4LUAD

2. Download the [BCSS-WSSS dataset](https://drive.google.com/drive/folders/1iS2Z0DsbACqGp7m6VDJbAcgzeXNEFr77) and put it in ./data/BCSS-WSSS (Thanks to [Han et. al](https://github.com/ChuHan89/WSSS-Tissue))

3. Download the [LUAD-HistoSeg dataset](https://drive.google.com/drive/folders/1E3Yei3Or3xJXukHIybZAgochxfn6FJpr) and put it in ./data/LUAD-HistoSeg (Thanks to [Han et. al](https://github.com/ChuHan89/WSSS-Tissue))

## Generate Synthesized Datasets

1. Synthesize datasets with Mosaic Transformation

    Run `./create_synthesis_datasets/mosaic_{wsss4luad|bcss|luad}.ipynb`

2. Synthesize datasets with Bézier Mask Generation

    Run `./create_synthesis_datasets/bezier_{wsss4luad|bcss|luad}.ipynb`

3. Train the synthesized image filtering module

    Run `./create_synthesis_datasets/discriminate_{wsss4luad|bcss|luad}.ipynb`

4. Obtain the filtered synthesized images

    #### For Mosaic Transformation (BCSS/LUAD-HistoSeg)

    ```bash
    CUDA_VISIBLE_DEVICES=0, python ./create_synthesis_datasets/filter_mosaic_{bcss|luad}.py --run 0
    ```

    #### For Mosaic Transformation (WSSS4LUAD)
    ```bash
    CUDA_VISIBLE_DEVICES=0, python ./create_synthesis_datasets/filter_mosaic_wsss4luad.py --run 0 --idx [0-9]
    ```

    NOTE: It takes some time to generate the filtered synthesized images for WSSS4LUAD. To accelerate the process, we utilize the `idx` argument enable multiple processing. 

    #### For Bézier Mask Generation

    ```bash
    CUDA_VISIBLE_DEVICES=0, python ./create_synthesis_datasets/filter_bezier_{wsss4luad|bcss|luad}.py --run 0
    ```

    

## Train the Segmentation Module

### Preparation for the WSSS4LUAD dataset

Since the Validation and Test images are not in the same shape for the WSSS4LUAD dataset, we first split them by a sliding window strategy with multi-scales. For the validation set, we utilize a sliding window size of `224` and a stride of `224`. For testing, we utilize a sliding window size of `224` and a stride of `112`. 

You can do the pre-processing by running `split_validation.ipynb`.

### Train & Test Scripts

Please check the `scripts` directory. For example, if you want to train on the WSSS4LUAD dataset, please run

```bash
bash scripts/run-wsss4luad.sh
```

### Reproduce the paper results
We tried our best to ensure the reproducibility of the results, but since the `torch.nn.functional.interpolate` function is **not deterministic**, the results may be different over runs if you train from scratch. If you want to fully reproduce the results, you can use the following Docker image with built-in weights on [Baidu Disk](https://pan.baidu.com/s/1J95TO3imscsHIx_pq1ahYQ?pwd=fjm9) (**code: fjm9**) or [OneDrive](https://1drv.ms/u/c/ed9170b4b8afd619/ESHgOGJ_gHhFiIfQgs9DX4MBfsiTsAAtl9iNz7m-yARBJw?e=6A8Bdi). And then run:
```
docker load < hisynseg_test.tar.gz

docker run --gpus "device=0" --rm -it --shm-size 8G -v /path/to/your/data:/opt/app/data -v /path/to/your/outputs:/opt/app/outputs hisynseg:test
```

Make sure you have give `777` access to the `./outputs` directory.


## Citation
If you find our work helpful, please cite our paper:
```text
@article{fang2024hisynseg,
  title={HisynSeg: Weakly-Supervised Histopathological Image Segmentation via Image-Mixing Synthesis and Consistency Regularization},
  author={Fang, Zijie and Wang, Yifeng and Xie, Peizhang and Wang, Zhi and Zhang, Yongbing},
  journal={IEEE Transactions on Medical Imaging},
  year={2024},
  publisher={IEEE}
}
```

 




