# DGPANet
This repo contains the code for our paper "Dual-Guided Prototype Alignment Network for Few-Shot Medical Image Segmentation''
![](https://github.com/shiyi0306/DGPANet/blob/main/DGPANet_framework.png?raw=true) 

### Dependencies
Please install following essential dependencies (see requirements.txt):
```
dcm2niix
json5==0.8.5
jupyter==1.0.0
nibabel==2.5.1
numpy==1.22.0
opencv-python==4.5.5.62
Pillow>=8.1.1
sacred==0.8.2
scikit-image==0.18.3
SimpleITK==1.2.3
torch==1.10.1
torchvision==0.11.2
tqdm==4.62.3
```

### Datasets and pre-processing
Download:  
1. **Abdominal MRI**  [Combined Healthy Abdominal Organ Segmentation dataset](https://chaos.grand-challenge.org/)  
2. **Cardiac MRI** [Multi-sequence Cardiac MRI Segmentation dataset (bSSFP fold)](https://zmiclab.github.io/zxh/0/mscmrseg19)  

**Pre-processing** is performed according to [Ouyang et al.](https://github.com/cheng-01037/Self-supervised-Fewshot-Medical-Image-Segmentation.git) and we follow the procedure on their github repository.  
We put the pre-processed images and their corresponding labels in `./data/CHAOST2/chaos_MR_T2_normalized` folder for Abdominal MRI and `./data/CMR/cmr_MR_normalized` folder for Cardiac MRI.  

**Supervoxel segmentation** is performed according to [Hansen et al.](https://github.com/sha168/ADNet.git) and we follow the procedure on their github repository.  
We also put the package `supervoxels` in `./data`, run our modified file `./data./supervoxels/generate_supervoxels.py` to implement pseudolabel generation. The generated supervoxels for `CHAOST2` and `CMR` datasets are put in `./data/CHAOST2/supervoxels_5000` folder and `./data/CMR/supervoxels_1000` folder, respectively.  

### Training  
1. Download pre-trained ResNet-101 weights [vanilla version](https://download.pytorch.org/models/resnet101-63fe2227.pth) or [deeplabv3 version](https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth) and put your checkpoints folder, then replace the absolute path in the code `./models/encoder.py`.  
2. Run `bash scripts/train_<abd,cmr>_mr.sh` 

### Testing
Run `bash scripts/val.sh`

### Acknowledgement
This code is based on [Q-Net](https://arxiv.org/abs/2208.11451) by [Shen et al.](https://github.com/ZJLAB-AMMI/Q-Net) and [SSP](https://arxiv.org/abs/2207.11549) ECCV 2022 by [Fan et al.](https://github.com/fanq15/SSP). 
