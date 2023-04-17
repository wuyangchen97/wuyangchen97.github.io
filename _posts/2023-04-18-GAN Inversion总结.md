---
layout:     post
title:      GAN Inversion总结
subtitle:   总结主要方法
date:       2023-04-18
author:     CWY
header-img: img/post-bg-cook.jpg
catalog: true
tags:
    - GAN
---

## 前言

- (1) directly optimize the latent vector to minimize the error for the given image.  
- (2) train an encoder to map the given image to the latent space.  
- (3) use a hybrid approach combining both.  


