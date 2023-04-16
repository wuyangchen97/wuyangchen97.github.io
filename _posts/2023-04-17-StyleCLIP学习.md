---
layout:     post
title:      StyleCLIP学习
subtitle:   主要算法与实验总结
date:       2023-04-17
author:     cwy
header-img: 
catalog: true
tags:
    - GAN
    - image-text
---
作者实际上提出了三种模型

## latent optimization. 
优化方法如下图所示：其中text就是希望图片能够修改成的target image，w是target image对应的latent code，也就是需要优化学习的参数，w_s是参考图片对应的latent code。
<img width="617" alt="image" src="https://user-images.githubusercontent.com/110716367/232315868-b9f2a1f2-17c3-4f4d-934c-dee3a844d440.png">

这种方式的缺点的每次manipulation需要一定时间（几分钟的优化过程）

## Latent Mapper
这种方式是构建一个专门的mapper模型，该模型能够学习到给出一个text，这个text对应在w+空间中的mannipulation step。


