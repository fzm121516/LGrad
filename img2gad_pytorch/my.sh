#!/bin/bash

# 获取当前脚本所在目录的绝对路径
GANmodelpath=$(cd $(dirname $0); pwd)/

# 从输入参数中获取图像根目录和保存目录
Imgrootdir=$2
Saverootdir=$3
Classes='0_real 1_fake'

# 定义测试集数据集类别
Testdatas='biggan deepfake gaugan stargan cyclegan/apple cyclegan/horse cyclegan/orange cyclegan/summer cyclegan/winter cyclegan/zebra progan/airplane progan/bicycle progan/bird progan/boat progan/bottle progan/bus progan/car progan/cat progan/chair progan/cow progan/diningtable progan/dog progan/horse progan/motorbike progan/person progan/pottedplant progan/sheep progan/sofa progan/train progan/tvmonitor stylegan/bedroom stylegan/car stylegan/cat stylegan2/car stylegan2/cat stylegan2/church stylegan2/horse'

# 定义测试集图像根目录和保存目录
Testrootdir=${Imgrootdir}
Savedir=$Saverootdir/test/

# 遍历每个测试集类别和类标签，生成梯度图像并保存
for Testdata in $Testdatas
do
    for Class in $Classes
    do
        Imgdir=${Testdata}/${Class}
        CUDA_VISIBLE_DEVICES=$1 python $GANmodelpath/gen_imggrad.py \
            ${Testrootdir}${Imgdir} \
            ${Savedir}${Imgdir}_grad \
            ./karras2019stylegan-bedrooms-256x256_discriminator.pth \
            1
    done
done
