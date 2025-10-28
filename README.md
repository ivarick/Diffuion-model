# Diffusion Model for Image Generation

A diffusion model implementation using U-Net architecture for generating high-quality 128×128 images.

## Overview

This project implements a diffusion probabilistic model that learns to generate images by iteratively denoising random noise. The model uses the U-Net architecture as its backbone, which has proven highly effective for image-to-image translation tasks and diffusion-based generation.

## Architecture

The model is built on the U-Net architecture, which features:

- **Encoder path**: Progressively downsamples the input through convolutional blocks, capturing hierarchical features
- **Bottleneck**: Processes the most compressed representation of the image
- **Decoder path**: Upsamples the features back to the original resolution
- **Skip connections**: Preserve fine-grained spatial information by connecting encoder and decoder layers directly

![U-Net Architecture](1.-UNet_What-Is-It-1.png)

*Image credit: [eviltux.com](https://eviltux.com/2024/08/11/training-a-u-net-model-from-scratch/)*

## How Diffusion Models Work

Diffusion models operate through two main processes:

1. **Forward diffusion**: Gradually adds noise to training images over multiple timesteps until they become pure random noise
2. **Reverse diffusion**: The U-Net learns to reverse this process, predicting and removing noise step-by-step to generate clean images from random noise

## Model Specifications

- **Output resolution**: 128×128 pixels
- **Architecture**: U-Net with time-step conditioning
- **Task**: Unconditional/conditional image generation
- **Training**: Dataset-agnostic (configurable for any image dataset)

## Key Features

- **Flexible dataset support**: Train on any image dataset by simply organizing images in the appropriate directory structure
- **Configurable generation**: Control the number of diffusion steps for quality vs. speed trade-offs
- **Scalable architecture**: Can be adapted for different image resolutions and conditioning methods

## Training Process

The model learns to:
1. Predict the noise added at each diffusion timestep
2. Gradually denoise random noise into coherent images
3. Capture the statistical distribution of the training dataset

## Use Cases

This diffusion model can be applied to:
- **Generative art**: Create novel images in the style of your training data
- **Data augmentation**: Generate synthetic training samples for downstream tasks
- **Image synthesis research**: Experiment with diffusion-based generation techniques
- **Custom dataset generation**: Train on specific domains (faces, landscapes, objects, etc.)


