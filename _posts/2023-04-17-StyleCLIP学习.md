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

## Latent optimization   
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
文中特别指出，CLIP空间中的图像manifold和文本manifold是两个不同的manifold，且没有一对一的关系。  
因为clip对特征是做了norm操作的，因此无法从特征长度上获得有用信息，文中给出的假设是：  
> 对于一个CLIP空间中的<image,text>对，给定同样的语义信息对它们改变时，它们在CLIP空间中的变化方向是近似共线的  

具体来说，其中 $\Delta t$ 是通过prompt engineering得到，例如目前操控的是一个人脸，目标是获得戴眼镜的人脸，那么$\Delta t$=embedding("face with glasses")-embedding("face")

整体示意图如下：  
<img width="664" alt="image" src="https://user-images.githubusercontent.com/110716367/232367200-8c658231-88e0-4131-b373-834226c219d0.png">  

为了达到图中的目的，则需要弄清Style空间中的s变化与CLIP空间中i变化的相关性。  
（这一块没有特别明白，可能有错误）  
做法是：每次只改变s中的某些entries，然后对改变后的s生成图片 $i+\Delta i = G(s+\Delta s)$，看该图片与原图$i=G(s)$在CLIP空间形成的$\Delta i$ 与 $\Delta t$ 的投影大小，如果大于设定阈值，则表明这一个channel是对结果有影响的。 

缺点（StyleGAN-NADA）：
> However, these approaches all share the limitation common to latent space editing methods - the modification that they can apply are largely constrained to the domain of the pre-trained generator. As such, they can allow for changes in hairstyle, expression, or even convert a wolf to a lion if the generator has seen both - but they cannot convert a photo to a painting, or produce cats when trained on dogs.  







