---
layout:     post
title:      Designing an Encoder for StyleGAN Image Manipulation学习
subtitle:   要点总结
date:       2023-04-17
author:     CWY
header-img: img/post-bg-cook.jpg
catalog: true
tags:
    - GAN
---

## 前言

这篇文章是StyleCLIP采用的编码器。

GAN iniversion 工作好坏的衡量标准：
- 图像重建效果要好  
- latent code的可编辑能力要强
  
作者指出，虽然之前的工作（如映射到w+空间）能重建大部分图片，但是存在的缺点是：
> inverting images away from the original W space reaches regions of the latent space that are less editable and in which the perceptual quality is lower.  

即没有很好的满足第二点要求