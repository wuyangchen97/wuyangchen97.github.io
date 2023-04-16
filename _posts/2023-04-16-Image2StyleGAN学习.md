---
layout:     post
title:      Image2StyleGAN学习
subtitle:   要点总结
date:       2023-04-16
author:     CWY
header-img: img/post-bg-cook.jpg
catalog: true
tags:
    - GAN
---

## 前言

最开始还不明白，为什么要有专门的模型/方法去做embedding，用判别器或分类器的前一层出来的feature vector不就是embedding了吗？用这个embedding也能实现下游的比对或检索任务。  
看了文章后，我认为该文章所指的embedding是指能够reconstruct original images的，这样才能将原图重新映射回embedding（不然只能不受掌控的由noise生成随机图案），而且需要embedding学习到的特征尽可能的彼此线性可分，这样能进一步在原图映射回的embedding上做编辑，编辑后再由生成网络得到图像，达到对原图的定向修改。

## embedding算法

<img width="440" alt="image" src="https://user-images.githubusercontent.com/110716367/232276737-01989358-f377-4834-9051-53a1812e3cc7.png">

如下
```python
def embedding_function(image):
    upsample = torch.nn.Upsample(scale_factor=256 / 1024, mode='bilinear')
    img_p = image.clone()
    img_p = upsample(img_p)
    perceptual = VGG16_perceptual().to(device)
    MSE_loss = nn.MSELoss(reduction="mean")  
    # W+ 
    
    latents = torch.zeros((1, 18, 512), requires_grad=True, device=device)
    optimizer = optim.Adam({latents}, lr=args.lr, betas=(0.9, 0.999), eps=1e-8)

    loss_ = []
    loss_psnr = []
    for e in range(args.epochs):#1500次即可  
        optimizer.zero_grad()
        syn_img = g_synthesis(latents)
        syn_img = (syn_img + 1.0) / 2.0  
	# original high-resolution real img --- MSE loss --- high-resolution synthesized img 
	
	# downsampled real img --- perceptual loss --- downsampled synthesized img  
	
	mse, per_loss = loss_function(syn_img, image, img_p, MSE_loss, upsample, perceptual)
        psnr = PSNR(mse, flag=0)
        loss = per_loss + mse
        loss.backward()
        optimizer.step()
```
其中的perceptual loss是采用vgg16,conv1_1,conv1_2,conv3_2,conv4_2的output  

## 论文要点
算法本身比较简单，主要是从实验去分析。
### 什么类型的图能够有效做embedding
测试方式：  
输入：faces of cat, dogs, and paintings； and register these images to a canonical face position （注意测试的两个要点：一是要共同享有face结构的；二是要配准至标准脸位置）  
模型：在human face上训练好的模型     
**结论：** 即使只在人脸类别上做了训练，但是模型依旧有泛化性，能对其他类别做embedding，知识恢复的效果有些模糊。如下图  
<img width="587" alt="image" src="https://user-images.githubusercontent.com/110716367/232279092-6d1a69ba-5721-4cef-b92c-f6974c7c5dc5.png">

### 图像变换/损坏对于embedding的影响
**结论1：** embedding算法对于原图的仿射变换非常敏感（其中平移的影响最大），如下图。
<img width="606" alt="image" src="https://user-images.githubusercontent.com/110716367/232279349-36f98927-04ef-4ffb-b40d-26d825dd6fcb.png">  

**结论2:** 但是对图像的defects非常鲁邦（这种现象有利于图像编辑方面的应用）。如下图。  
<img width="301" alt="image" src="https://user-images.githubusercontent.com/110716367/232279427-7999834b-e0e9-4b6d-8f6e-f38743f09659.png">


### 选择什么空间做embedding
stylegan中是`z->mappinng net->w->synthesize net->image`
可以选择的隐空间有最开始的noise z，中间输出w。  
但是文中说，这两种的效果都不是很好，最终选择了w+空间，也就是中间输出w的扩展。w的shape是(1,512),而w+则是(18,512),其中的18对应了synthesize net的每一层。  
此外，文中还进行了额外的实验，看synthesize net的权重是否会影响重建的效果，结果如下图，能发现w+空间得到的(f),(g)明显好于w空间的(c),(d)  
> 这儿没太明白：个人推测是用训练好的网络优化得到embedding后，再随机初始化synthesize net的权重，看生成图像的质量).  
     
<img width="608" alt="image" src="https://user-images.githubusercontent.com/110716367/232280171-61ef6b02-5e9c-4bd6-855a-afc7ef1b9c7c.png">   


### embedding的意义

- Morphing  
简单来说就是对两张图的embedding做加权后，生成新的图片。
$$ w=\lambda w_1 + (1-\lambda)w_2 $$  
实验发现，对于不是人脸的图像来说，Morphing的效果不好。    
<img width="291" alt="image" src="https://user-images.githubusercontent.com/110716367/232280470-e011a2e7-ef37-4d1c-ad5e-c2da27e4e1a1.png">      

```python
def morphing(w0, w1, img_id0, img_id1):
    '''
        morphing operation
    '''
    for i in range(20):
        a = (1 / 20) * i
        w = w0 * (1 - a) + w1 * a
        syn_img = g_synthesis(w)
        syn_img = (syn_img + 1.0) / 2.0
        save_image(syn_img.clamp(0, 1), "save_images/image2stylegan/morphing/morphed_{}_{}_{}.png".format(img_id0, img_id1, i))
```

- Style Transfer  
用w1前9个vector的作为content，用w2的后9个vector作为style，合并为w   
<img width="295" alt="image" src="https://user-images.githubusercontent.com/110716367/232283337-87b34d19-a0a2-45c4-8874-c9c496555774.png"> 

> 这儿没有懂图中写的style loss是什么意思 

```python
def style_transfer(target_latent, style_latent, src, tgt):
    '''
        style transfer
    '''
    tmp_latent1 = target_latent[:, :10, :]
    tmp_latent2 = style_latent[:, 10:, :]
    latent = torch.cat((tmp_latent1, tmp_latent2), dim=1)
    print(latent.shape)
    syn_img = g_synthesis(latent)
    syn_img = (syn_img + 1.0) / 2.0
    save_image(syn_img.clamp(0, 1), "save_images/image2stylegan/style_transfer/Style_transfer_{}_{}_10.png".format(src, tgt))
```

- Expression Transfer and Face Reenactment  
简单来说就是用两张图的表情差异的到$\delta embedding$,然后对目标图像的embedding进行修改。  
$$w=w_1 + \lambda(w_3-w_2)$$ 
其中w2的图像一般是自然的表情，w3是目标表情


### 其他
有意思的现象:对于latent的初始化，作者发现，若采用w的均值作为初始化值，那么人脸上的效果好，但是其他类别反而不好。不是人脸的类别采用uniform distribution初始化反而更好，因此猜测，这其中的原因为：
>  Intuitively, the phenomenon suggests that the distribution has only one cluster of faces, the other instances (e.g. dogs, cats) are scattered points surrounding the cluster without obvious patterns. 



### 参考
- https://github.com/Jerry2398/Image2StyleGAN-and-Image2StyleGAN-

