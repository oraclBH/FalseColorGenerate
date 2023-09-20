'''
author:BH
data:2023/09/20
'''

# Gray image false color generation, low gray value corresponds to purple,
# high gray value corresponds to yellow


import os
import sys
import numpy as np
import matplotlib.pyplot as plt

'''
param :level ---- 8bit数据取值范围[0, 255], level = 256, 10bit数据取值范围[0, 1023], level=1024
param :purpleToblue 根据色相环表，紫到蓝色， blueToCyan 蓝到青 cyanTogreen 青到绿 greenToyellow 绿到黄
'''

#显示生成的色相图
def showMap(map):
    mapFigure = np.zeros((500, len(map), 3), dtype=np.float)
    for i in range(len(map)):
        mapFigure[:, i, :] = np.repeat([map[i, :]], 500, axis=0)
    plt.figure()
    plt.imshow(np.uint8(mapFigure))
    plt.show()

# 生成色相图
def adaptionColorMap(level, purpleToblue=0.25, blueToCyan=0.25, cyanTogreen=0.25, greenToyellow=0.25):
    block = [level*purpleToblue, level*blueToCyan, level*cyanTogreen, level*greenToyellow]
    # 取整数
    block = [round(x) for x in block]
    sum_block = sum(block)

    #保证block中数值累加等于level
    if sum_block is not level:
        block[0] += (level - sum_block)

    #构建映射表
    map = np.zeros((level, 3), dtype=np.float)

    # purpleToblue
    map[0:block[0], 0] = np.linspace(block[0], 1, block[0]) / block[0]
    map[0:block[0], 1] = np.zeros(block[0], dtype=np.float)
    map[0:block[0], 2] = np.ones(block[0], dtype=np.float)

    # blueToCyan
    map[block[0]:block[1]+block[0], 0] = np.zeros(block[1], dtype=np.float)
    map[block[0]:block[1] + block[0], 1] = np.linspace(1, block[1], block[1]) / block[1]
    map[block[0]:block[1] + block[0], 2] = np.ones(block[1], dtype=np.float)

    # cyanTogreen
    map[block[0]+block[1]:block[2]+block[1]+block[0], 0] = np.zeros(block[2], dtype=np.float)
    map[block[0]+block[1]:block[2]+block[1]+block[0], 1] = np.ones(block[2], dtype=np.float)
    map[block[0]+block[1]:block[2]+block[1]+block[0], 2] = np.linspace(block[2], 1, block[2]) / block[2]

    # greenToyellow
    map[block[0]+block[1]+block[2]:block[3]+block[2]+block[1]+block[0], 0] = np.linspace(1, block[3], block[3]) / block[3]
    map[block[0]+block[1]+block[2]:block[3]+block[2]+block[1]+block[0], 1] = np.ones(block[3], dtype=np.float)
    map[block[0]+block[1]+block[2]:block[3]+block[2]+block[1]+block[0], 2] = np.zeros(block[3], dtype=np.float)

    #数值统一到[0, level-1]之间并取整
    map = np.uint8(map*(level-1))
    return map

#灰度图像映射成图像
def generateFalseColorImage(gray, map):
    m, n = gray.shape
    colorImg = np.zeros((m, n, 3), dtype=np.uint8)
    for i in range(m):
        for j in range(n):
            v = gray[i, j]
            colorImg[i, j, :] = map[v, :]
    return colorImg


if __name__ == "__main__":
    # IM = plt.imread('man.bmp')
    IM = plt.imread('1234.bmp')
    IM_gray = (0.2989 * IM[:,:,0] + 0.5870 * IM[:,:,1] + 0.1140 * IM[:,:,2]).astype(np.uint8)
    plt.figure(1)
    plt.imshow(IM_gray, cmap='gray')
    plt.title('gray-image')

    map = adaptionColorMap(256)
    falseIm = generateFalseColorImage(IM_gray, map)
    plt.figure(2)
    plt.imshow(falseIm)
    plt.title('falseColor-image')

    plt.figure(3)

    IM_GRAY = np.zeros(falseIm.shape, dtype=np.uint8)
    for i in range(IM_GRAY.shape[0]):
        for j in range(IM_GRAY.shape[1]):
            IM_GRAY[i, j, :] = np.repeat([IM_gray[i, j]], 3, axis=0)

    plt.imshow(np.concatenate((IM_GRAY, falseIm), axis=1))
    plt.title('gray-falseColor-image-showpair')
    plt.show()



