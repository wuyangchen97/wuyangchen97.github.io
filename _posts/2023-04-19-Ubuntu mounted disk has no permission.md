---
layout:     post
title:      Mounted disk has no permission
subtitle:   record the solution
date:       2023-04-19
author:     CWY
header-img: img/post-bg-cook.jpg
catalog: true
tags:
    - Ubuntu
---

## BackGround
Assuming you have a new disk that needs to be mounted, you can refer to following steps

## Solution
step1: check the filesystem and other informations 
`lsblk -f`  

step2: get the user id(uid=xx in step3):  
`grep ^"$USER" /etc/group`

step3: mount it using the following command: 

> with execute permissions for files, no access for 'others'  

`sudo mount -o rw,user,uid=1000,umask=007,exec /dev/sdxn /mnt/sd1  # general syntax`    

!note **donot use below command**, as it would cause `git clone` to fail
> Cloning into 'xx'...  
error: chmod on /mnt/xx/.git/config.lock failed: Operation not permitted  
fatal: could not set 'core.filemode' to 'false'    
 
use `git global --config core.filemode` false not work

references:

https://askubuntu.com/questions/11840/how-do-i-use-chmod-on-an-ntfs-or-fat32-partition/956072#956072

https://www.jianshu.com/p/3b0a9904daca


