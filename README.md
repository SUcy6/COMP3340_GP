# README

# COMP3340_Group16_GP 🏆

### **Connecting low-level image processing and high-level vision for degraded image classification**

### 📋Progress:

Project Proposal on Sep 16: [proposal](https://github.com/SUcy6/COMP3340_GP/blob/main/COMP3340_Group%2016_Proposal.pdf)

Midterm Report on Oct 30:

Final presentation slides:

Final report:

## 📊Datasets

[17 Category Flower Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/17/): a 17 category flower dataset with 80 images for each class

![flower17.png](README%2083d2b46279f74e17b041618af732df83/flower17.png)

[Benchmarking Neural Network Robustness to Common Corruptions and Perturbations](https://github.com/hendrycks/robustness): Tiny ImageNet-C, CIFAR-10-C

![imagenet-c.png](README%2083d2b46279f74e17b041618af732df83/imagenet-c.png)

### Data Prepare

**For Oxford flower17:**

1. Download the images from [17 Category Flower Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/17/)
2. Extract the file and get a **jpg** folder with all the 1360 images
3. Build a new data folder containing **17 subfolders** (for categories 1 to 17)
4. separate and distribute the images to the **17 subfolders** (each 80 images is one category in Flower17)

flowers.csv: a helper csv file for distribute the images

## 📌Experiment Result

We output the training loss, validation loss, and test accuracy to csv file. From these files, we plot the graphs to visualize our experiment results.

All the output csv can be found [here](https://github.com/SUcy6/COMP3340_GP/tree/main/output).

![截屏2022-10-27 下午1.08.09.png](README%2083d2b46279f74e17b041618af732df83/%25E6%2588%25AA%25E5%25B1%258F2022-10-27_%25E4%25B8%258B%25E5%258D%25881.08.09.png)

## 🏠Model Architecture

The detailed model architecture and training process can be found [here](https://github.com/SUcy6/COMP3340_GP/tree/main/Model).

To reproduce our experiment results, please download the jupyter notebooks and train on your cloud with GPU.

## 🌗Low-quality image classification

In the second part of our project, we decide to test the baseline models on degraded image dataset. After that, we will try to implement the newly proposed method for low-quality image classification. They successfully combine low-level image processing with high level classification task.

****Dirty Pixels: Towards End-to-End Image Processing and Perception****

[https://github.com/princeton-computational-imaging/DirtyPixels](https://github.com/princeton-computational-imaging/DirtyPixels)

**WaveCNet: Wavelet Integrated CNNs to Suppress Aliasing Effect for Noise-Robust Image Classification**

[https://github.com/LiQiufu/WaveCNet](https://github.com/LiQiufu/WaveCNet)

[https://paperswithcode.com/paper/wavecnet-wavelet-integrated-cnns-to-suppress](https://paperswithcode.com/paper/wavecnet-wavelet-integrated-cnns-to-suppress)

**Group-wise Inhibition based Feature Regularization for Robust Classification**

[https://paperswithcode.com/paper/group-wise-inhibition-based-feature](https://paperswithcode.com/paper/group-wise-inhibition-based-feature)

[https://github.com/LinusWu/TENET_Training](https://github.com/LinusWu/TENET_Training)

****Robust Image Classification Using A Low-Pass Activation Function and DCT Augmentation****

[https://github.com/tahmid0007/Low_Pass_ReLU](https://github.com/tahmid0007/Low_Pass_ReLU)

**When Image Denoising Meets High-Level Vision Tasks: A Deep Learning Approach**

[https://arxiv.org/pdf/1706.04284.pdf](https://arxiv.org/pdf/1706.04284.pdf)

## 🗓️Midterm Plan

- Build the baseline models
- training
- Test and evaluate models(plot accuracy/loss, test result)
- Report writting (further explanation)

## 📝Final Plan

## 🤡Tutorials

[alexnet on celeba](https://www.youtube.com/watch?v=6c8WFGbPHpE)

[resnet from scratch](https://www.youtube.com/watch?v=DkNIBBBvcPs)

[pytorch pretrained models](https://www.youtube.com/watch?v=qaDe0qQZ5AQ&t=14s)

[finetuning pytorch pretrained](https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html)