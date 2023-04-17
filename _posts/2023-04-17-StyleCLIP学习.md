---
layout:     post
title:      StyleCLIP学习
subtitle:   主要算法与实验总结
date:       2023-04-17
author:     cwy
header-img: img/post-bg-cook.jpg
catalog: true
tags:
    - GAN
    - MultiModal
---

## 前沿
StyleCLIP的目的是，输入一张图像与对应的文字描述，期望该图像能够按照文字的描述进行变化，其余特征保持不变。  
作者针对该任务实际上提出了三种模型。

## Latent optimization. 
优化方法如下图所示：其中text就是希望图片能够修改成的target image，w是target image对应的latent code，也就是需要优化学习的参数，w_s是参考图片对应的latent code。  
<img width="582" alt="image" src="https://user-images.githubusercontent.com/110716367/232316469-b3beb434-f16b-4727-8bf2-92becb52863e.png">

这种方式的缺点:  
- 每次manipulation需要一定时间（几分钟的优化过程）  
- 对于超参数比较敏感 

## Latent Mapper
这种方式是构建一个专门的mapper模型，该模型能够学习到给出一个text，这个text对应在w+空间中的manipulation step。  
如下图所示，学习到的东西可以理解为一个残差residual。 
> 但这这个mapper好像是固定了<image,text> pair，即给定一张图像，只能是训练集中固定的这个text进行编辑    

<img width="799" alt="image" src="https://user-images.githubusercontent.com/110716367/232319477-2a0ba9b7-5a96-4f91-a4c3-7940c0b55835.png">   

优化方法如下图所示：  
<img width="831" alt="image" src="https://user-images.githubusercontent.com/110716367/232360252-7150d5d2-f12b-48a2-bd66-93cf1a536b92.png">  

这种方式的缺点：  
- falls short when a fine-grained disentangled manipulation is desired.  
- the directions of different manipulation steps for a given text prompt tend to be similar.  

## Gobal Direction  
TODO. 




