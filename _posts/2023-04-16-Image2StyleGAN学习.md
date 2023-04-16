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
    for e in range(args.epochs):
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

## 论文要点

`Perceptual Loss`，一种非正式协议，通知其他对象的指定属性发生了改变。

简单理解就是，监听一个对象的某个`属性`是否发生改变。

### 关于latent space空间的选择

- 最开始的noise，Z空间

```	objc
- (void)addObserver:(NSObject *)observer forKeyPath:(NSString *)keyPath options:(NSKeyValueObservingOptions)options context:(nullable void *)context;
```


- 经过全连接映射后的W空间

```objc
- (void)observeValueForKeyPath:(nullable NSString *)keyPath ofObject:(nullable id)object change:(nullable NSDictionary<NSKeyValueChangeKey, id> *)change context:(nullable void *)context;
```

- W+空间  
文中说“An important insight of our work is that it is not easily possible to embed into W or Z directly",
因此将W扩展为了W+空间，该空间维度为(18,512),18分别对应g_synthesize生成网络的每一层

```objc
- (void)removeObserver:(NSObject *)observer forKeyPath:(NSString *)keyPath;
```


代码演示

```objc
- (void)viewDidLoad {
    [super viewDidLoad];
    
    self.personModel = [[BYPersonModel alloc] init];
    [self.personModel setName:@"Tony Qiu"];
    
    /// 添加监听
    /// options:NSKeyValueObservingOptionNew | NSKeyValueObservingOptionOld 监听新值和旧值,若不传则在监听方法中，无法捕获变化的值
    [self.personModel addObserver:self forKeyPath:@"name" options:NSKeyValueObservingOptionNew | NSKeyValueObservingOptionOld context:nil];
    [self.personModel addObserver:self forKeyPath:@"age" options:NSKeyValueObservingOptionNew context:nil];
    
    /// 改变属性值 就能在监听中捕获变化
	[self.personModel setName:@"Peng YuYan"];
    [self.personModel setAge:28];
}

/// 在非正式协议里监听对象变化
- (void)observeValueForKeyPath:(NSString *)keyPath ofObject:(id)object change:(NSDictionary<NSKeyValueChangeKey,id> *)change context:(void *)context{
    NSLog(@"%@", change);
}

/// 移除监听
- (void)dealloc {
    [self.personModel removeObserver:self forKeyPath:@"name"];
    [self.personModel removeObserver:self forKeyPath:@"age"];
}


```

输出

```
2021-03-19 14:35:02.814222+0800 KVO_demo[34947:1626934] {
    kind = 1;
    new = "Peng YuYan";
    old = "Tony Qiu";
}
2021-03-19 14:35:02.814448+0800 KVO_demo[34947:1626934] {
    kind = 1;
    new = 28;
}
```


### KVO底层实现

首先，我们用`runtime`在添加监听之前和之后分别打印一下类对象


```objc
NSLog(@"%@", object_getClass(self.personModel));
[self.personModel addObserver:self forKeyPath:@"name" options:NSKeyValueObservingOptionNew | NSKeyValueObservingOptionOld context:nil];
NSLog(@"%@", object_getClass(self.personModel));
```

```
KVO_demo[75775:1761189] BYPersonModel
KVO_demo[75775:1761189] NSKVONotifying_BYPersonModel

```
也可以在 `lldb` 中打印, 
> 不能打印 [self.personModel class],后面会说到为什么

```
(lldb) po self.personModel.isa
BYPersonModel

  Fix-it applied, fixed expression was: 
    self.personModel->isa
(lldb) po self.personModel.isa
NSKVONotifying_BYPersonModel

  Fix-it applied, fixed expression was: 
    self.personModel->isa
(lldb) 
```

会发现添加监听后的`personModel`的类从 `BYPersonModel` 变成了`NSKVONotifying_BYPersonModel`，也就是`NSKVONotifying_+类名`的形式。
就是说系统为我们自动生创建了一个新的类，然后通过这个类去实现监听方法。

进一步验证，我们自己创建一个`NSKVONotifying_BYPersonModel`类，添加KVO时，会发出警告

```
KVO_demo[19623:258692] BYPersonModel
KVO_demo[19623:258692] [general] KVO failed to allocate class pair for name NSKVONotifying_BYPersonModel, automatic key-value observing will not work for this class
KVO_demo[19623:258692] BYPersonModel
```

并且系统无法自动生成`NSKVONotifying_BYPersonModel`类。

下面我们使用下面打印`NSKVONotifying_BYPersonModel`的属性和方法

```objc
/// 打印方法
- (void)methodsByClass:(Class)cls{
    NSLog(@"%@ methods:", cls);
    unsigned int count;
    Method *methods = class_copyMethodList(cls, &count);
    
    for (NSInteger index = 0; index < count; index++) {
        Method method = methods[index];
        
        NSString *methodStr = NSStringFromSelector(method_getName(method));
        NSLog(@"%@", methodStr);
    }
    
    free(methods);
}

/// 打印属性
- (void)ivarsByClass:(Class)cls{
    NSLog(@"%@ ivars:", cls);
    unsigned int count;
    Ivar *ivars = class_copyIvarList(cls, &count);
    
    for (NSInteger index = 0; index < count; index++) {
        Ivar ivar = ivars[index];
        NSString *ivarName = [NSString stringWithUTF8String:ivar_getName(ivar)];  //获取成员变量的名字
        NSString *ivarType = [NSString stringWithUTF8String:ivar_getTypeEncoding(ivar)]; //获取成员变量的数据类型
        NSLog(@"%@ %@", ivarName, ivarType);
    }
    
    free(ivars);

}
```

输出

```
KVO_demo[16813:215245] NSKVONotifying_BYPersonModel methods:
KVO_demo[16813:215245] setName:
KVO_demo[16813:215245] class
KVO_demo[16813:215245] dealloc
KVO_demo[16813:215245] _isKVOA
KVO_demo[16813:215245] NSKVONotifying_BYPersonModel ivars:

```

观察可以发现 `NSKVONotifying_BYPersonModel` 没有`ivar`。
重写了`setName `、`class `和`dealloc `方法，还新增了一个`_isKVOA`方法
-  `_isKVOA`用来判断是否是系统生成的`KVO`
-  `setName:`重写Set方法，并发送监听
-  `class` 返回父类，隐藏系统生成的 `NSKVONotifying_类`
-  `dealloc`销毁时移除一些方法

#### 我们来看看重写的`set`方法做了什么

打断点，用户 `lldb`打印 `KVO`前后的 `setName:`方法

```
(lldb) p [self.personModel methodForSelector:@selector(setName:)]
(IMP) $1 = 0x00000001059aef00 (KVO_demo`-[BYPersonModel setName:] at BYPersonModel.h:14)
(lldb) p [self.personModel methodForSelector:@selector(setName:)]
(IMP) $2 = 0x00007fff207d2583 (Foundation`_NSSetObjectValueAndNotify)
```

首先可以发现`setName:`方法的指针指向变了，从`[BYPersonModel setName:]`指向了 `Foundation `的 `_NSSetObjectValueAndNotify`的C语言方法


`_NSSetObjectValueAndNotify`内部做了什么呢？通过越狱手机可以获取`Foundation`框架，使用`Hopper`来解析源码生成的是汇编语言，看看汇编源码会发现`_NSSetObjectValueAndNotify`内部注释有提示说调用`didChangeValueForKey`

#### 尝试手动触发一个KVO

```
- (void)touchesBegan:(NSSet<UITouch *> *)touches withEvent:(UIEvent *)event {
    [self.personModel willChangeValueForKey:@"name"];
    [self.personModel didChangeValueForKey:@"name"];
}
```

直接调用 `willChangeValueForKey:`和 `didChangeValueForKey:`后，能发现触发了 `KVO `的`observeValueForKeyPath`方法。
单独调用`willChangeValueForKey:`或`didChangeValueForKey:`，则不会触发。

这就验证了`_NSSetObjectValueAndNotify` 的一些内部操作。

#### 到此整个`KVO`流程基本上就清晰了：

![](https://user-gold-cdn.xitu.io/2018/9/21/165fb89bc6ba9262?w=1878&h=898&f=png&s=150722)





### 动态生成一个自己的类

通过 `KVO` 底层的学习，我们知道了如何动态生成一个自己的类。


```objc
- (void)creatClass {
	/// 创建类
	Class customClass = objc_allocateClassPair([NSObject class], "BYCustomClass", 0);
	/// 添加实例变量和方法
	class_addIvar(customClass, "age", sizeof(int), 0, "i");
	class_addIvar(customClass, "name", sizeof(id), log2(sizeof(id)), @encode(id));
	/// 添加方法，`V@:`表示方法的参数和返回值
	class_addMethod(customClass, @selector(gohome), (IMP)gohome, "V@:");
	/// 注册到运行时环境(注意：注册后无法再添加方法和实例变量)
	objc_registerClassPair(customClass);
}

void gohome(id self, SEL _cmd)
{
    NSLog(@"回家了");
}

- (void)gohome {
}
```


### 自己写一个KVO

`KVO`的原理知道了，我们尝试自己写一个`KVO`吧

```objc
@interface NSObject (kvo)
/// 添加一个KVO方法
- (void)by_addObserver:(NSObject *)observer forKeyPath:(NSString *)keyPath options:(NSKeyValueObservingOptions)options context:(nullable void *)context;
@end
```
> 下面代码运行会报错 在 `Build Settings` 中设置`ENABLE_STRICT_OBJC_MSGSEND = NO`即可


```objc
#import "NSObject+kvo.h"
#import <objc/runtime.h>
#import <objc/message.h>

@implementation NSObject (kvo)

- (void)by_addObserver:(NSObject *)observer forKeyPath:(NSString *)keyPath options:(NSKeyValueObservingOptions)options context:(void *)context{
    //动态添加一个类
    NSString *originClassName = NSStringFromClass([self class]);
    
    NSString *newClassName = [@"BY_NSKVONotifying_" stringByAppendingString:originClassName];
    
    // 继承自当前类，创建一个子类，类名模仿KVO底层命名 BY_NSKVONotifying_+类名的形式
    Class kvoClass = objc_allocateClassPair([self class], [newClassName UTF8String], 0);
    
    // 添加setter方法 这里我们只监听 name，手动添加setName方法。
    // v@:@：v 对应setName方法的返回值void，@: 表示方法本身，@ 表示参数是个对象
    class_addMethod(kvoClass, @selector(setName:), (IMP)setName, "v@:@");
    
    //注册新添加的这个类
    objc_registerClassPair(kvoClass);
    
    // 修改isa指针，由 personModel 指向我们创建的 BY_NSKVONotifying_BYPrsonModel 对象实现替换
    object_setClass(self, kvoClass);
    
    // 保存观察者属性到当前类中
    objc_setAssociatedObject(self, (__bridge const void *)@"observer", observer, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
}

#pragma mark - 重写父类方法

void setName(id self, SEL _cmd, NSString *name) {
    
    // 保存当前KVO的类
    Class kvoClass = [self class];
    
    // 将self的isa指针指向父类BYPersonModel，调用父类setter方法
    object_setClass(self, class_getSuperclass([self class]));
    objc_msgSend(self, @selector(setName:), name);
    
    // 获取BY_NSKVONotifying_BYPrsonModel观察者
    id objc = objc_getAssociatedObject(self, (__bridge const void *)@"observer");
    // 通知观察者，执行通知方法
    NSDictionary<NSKeyValueChangeKey,id> *change = @{@"kind": @1, @"new": name};
    objc_msgSend(objc, @selector(observeValueForKeyPath:ofObject:change:context:), @"name", self, change, nil);
    
    // 将指针重新指向 BY_NSKVONotifying_BYPrsonModel
    object_setClass(self, kvoClass);
}


@end
```

使用我们的kvo方法

```objc
self.personModel = [[BYPersonModel alloc] init];
[self.personModel setName:@"Tony Qiu"];
    
NSLog(@"%@", object_getClass(self.personModel));
[self.personModel by_addObserver:self forKeyPath:@"name" options:NSKeyValueObservingOptionNew | NSKeyValueObservingOptionOld context:nil];
NSLog(@"%@", object_getClass(self.personModel));
```

输出

```
KVO_demo[51608:1429122] BYPersonModel
KVO_demo[51608:1429122] BY_NSKVONotifying_BYPersonModel
KVO_demo[51608:1429122] {
    kind = 1;
    new = "Peng YuYan";
}
```

可以看到，`BYPersonModel`类被替换成了`BY_NSKVONotifying_BYPersonModel`类，也能监听到`name`的变化，手写KVO成功。
当然实际的KVO实现的细节远比我们手写的复杂，这个只是一探究竟而已。



### 参考
- [《Introduction to Key-Value Observing Programming Guide》
](https://developer.apple.com/library/archive/documentation/Cocoa/Conceptual/KeyValueObserving/KeyValueObserving.html#//apple_ref/doc/uid/10000177-BCICJDHA)
- https://blog.csdn.net/science_lee/article/details/82843080
- https://www.cenzhijun.top/2018/05/kvo/
