# Satellite Image Segmentation with Light U-Net

A TensorFlow implementation of a lightweight U-Net style semantic segmentation framework for satellite and remote sensing imagery.

This project focuses on pixel-wise segmentation of aerial images and demonstrates a practical deep learning workflow for remote sensing image analysis.

## Overview

Semantic segmentation is an important task in remote sensing, where each pixel in an image is assigned to a semantic class such as **road**, **building**, **water**, or **background**.

This repository implements a **Light U-Net** framework for satellite image segmentation using **TensorFlow 1.x**. The architecture combines efficient encoder-decoder design with residual-style blocks and pyramid-style upsampling to produce detailed segmentation masks.

The original framework was used in the **2017 CCF BDCI Remote Sensing Image Semantic Segmentation Challenge** and reportedly achieved **0.891 accuracy**.

## Key Features

- Lightweight U-Net style segmentation framework
- Designed for remote sensing and satellite image analysis
- TensorFlow-based training and inference pipeline
- Residual-style convolution blocks
- Pyramid-like upsampling structure
- Example qualitative results included

## Preview

<p align="center">
  <img src="/sample_visible.png" alt="Satellite Input Image" width="40%">
  <img src="/sample_result.png" alt="Segmentation Result" width="40%">
</p>

## Environment

This project was originally developed with the following environment:

- Ubuntu 16.04
- Python 2.7
- TensorFlow 1.3
- OpenCV 3.2
- CUDA 8.0

> This is a legacy TensorFlow 1.x implementation. A compatible NVIDIA GPU environment is recommended for training and testing.

## Repository Structure

- `Network.py` — network architecture
- `dataset-processing.py` — dataset preparation and preprocessing
- `train.py` — model training
- `test-model.py` — inference / testing
- `utils.py` — helper functions
- `sample_visible.png` — example input image
- `sample_result.png` — example segmentation output
- `LICENSE`
- `README.md`

## Network Architecture

The model follows a **Light U-Net / encoder-decoder segmentation design** with the following characteristics:

- Feature pyramid style architecture
- Linear interpolation for upsampling instead of deconvolution
- Residual-style convolution blocks
- Downsampling through stride-based convolution
- Conditional Random Field (CRF) refinement at the network output
- Softmax cross-entropy loss for segmentation training

This design aims to balance **segmentation quality** and **computational efficiency** for satellite image understanding tasks.

## Dataset

The project is based on the **2017 CCF BDCI Remote Sensing Image Segmentation Challenge** dataset.

Original training images and labels are provided in PNG format, where each pixel corresponds to a semantic category.

Example classes include:

- background
- road
- building
- water
- plane

Two label settings were described in the original implementation:

### BDCI-jiage
- plane (1)
- road (2)
- building (3)
- water (4)
- other (0)

### BDCI-jiage-Semi
- plane (1)
- building (2)
- water (3)
- road (4)
- other (0)

To generate the training data:

- random **1024 × 1024** patches are selected from the original maps
- patches are resized to **256 × 256**
- augmentation includes rotations of **0°, 90°, 180°, 270°** and mirror transformations

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Ilyas-Raza1214/satellite-segmentation-Unet.git
cd satellite-segmentation-Unet
```

### 2. Install dependencies

This project uses legacy TensorFlow 1.x and OpenCV-based dependencies.

You may also need to install **PyDenseCRF** separately:

```bash
git clone https://github.com/lucasb-eyer/pydensecrf.git
cd pydensecrf
python setup.py install
```

### 3. Download dataset and pretrained model

Download the dataset and pretrained model from the original source if available, then place them inside the project directory.

After downloading:

- unzip the package into the repository folder
- rename the checkpoint file to:

`UNet_ResNet_itr100000.ckpt`

## Data Preparation

To generate the TFRecord dataset, run:

```bash
python dataset-processing.py
```

> Make sure the dataset paths inside the script are correctly configured before running preprocessing.

## Training

Before training, review the dataset paths, training parameters, and GPU settings in the code.

Run training with:

```bash
python train.py --gpu=0
```

Trained checkpoints will be saved in the model directory with names similar to:

`UNet_ResNet_itrxxxxxx.ckpt`

## Testing

A pretrained model checkpoint such as `UNet_ResNet_itr100000.ckpt` can be used for inference.

Run testing with:

```bash
python test-model.py --gpu=0
```

The generated segmentation results will be saved to the output directory specified in the testing script.

## Results

The repository includes example qualitative results for satellite image segmentation.

Typical outputs include:

- original satellite image
- predicted segmentation mask
- refined segmentation results

You can place additional visual examples in the repository to better showcase model performance.

## Applications

This type of segmentation framework can be useful for:

- land-cover analysis
- road extraction
- building segmentation
- water body detection
- remote sensing image interpretation
- environmental and urban planning analysis

## Limitations

- Built on **Python 2.7** and **TensorFlow 1.3**
- Requires a legacy environment for reproducibility
- May need code modernization for current TensorFlow or PyTorch ecosystems
- Dataset download and preprocessing paths may need manual adjustment

## Future Improvements

Possible future improvements for this repository include:

- migrate code to **Python 3**
- update framework to **TensorFlow 2.x** or **PyTorch**
- add a `requirements.txt` file
- provide a notebook demo
- include evaluation metrics such as IoU and Dice score
- add more visual examples and experiment logs
- improve reproducibility with clearer dataset setup instructions

## References

1. Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun,  
   **Deep Residual Learning for Image Recognition**  
   https://arxiv.org/abs/1512.03385

2. Tsung-Yi Lin, Piotr Dollár, Ross Girshick, Kaiming He, Bharath Hariharan, Serge Belongie,  
   **Feature Pyramid Networks for Object Detection**  
   https://arxiv.org/abs/1612.03144

3. Olaf Ronneberger, Philipp Fischer, Thomas Brox,  
   **U-Net: Convolutional Networks for Biomedical Image Segmentation**  
   https://arxiv.org/abs/1505.04597

## Acknowledgment

This repository is based on or adapted from earlier work on satellite image segmentation using Light U-Net / FPN-style design for the 2017 CCF BDCI remote sensing challenge.

If you are reusing or extending an existing implementation, it is good practice to keep proper attribution to the original authors and repository.

## Author

**Muhammad Ilyas Raza**  
Machine Learning Engineer | Computer Vision | Robotics | Applied AI

GitHub: [Ilyas-Raza1214](https://github.com/Ilyas-Raza1214)

## License

This project is distributed under the license included in this repository.
