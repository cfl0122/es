# -*- coding: utf-8 -*-
"""
Created on 2019‎年‎11‎月‎20‎日
此模块为计算重叠长度的模块
@author: hejieheng
"""
import numpy as np
import math


def cal_overlop(wall1, wall2, angle, center = [0,0]):
    wall1 = np.reshape(wall1, (-1, 2))
    wall2 = np.reshape(wall2, (-1, 2))
    theta = np.pi * angle / 180  # 以顺时针旋转为正
    # 旋转两个楼栋
    wall1_lis = list()
    for i in range(len(wall1)):
        x0, y0 = wall1[i]
        new_x = (x0 - center[0]) * math.cos(theta) + (y0 - center[1]) * math.sin(theta) + center[0]
        new_y = -(x0 - center[0]) * math.sin(theta) + (y0 - center[1]) * math.cos(theta) + center[1]
        wall1_lis.append([round(new_x, 3), round(new_y, 3)])
    wall2_lis = list()
    for i in range(len(wall2)):
        x0, y0 = wall2[i]
        new_x = (x0 - center[0]) * math.cos(theta) + (y0 - center[1]) * math.sin(theta) + center[0]
        new_y = -(x0 - center[0]) * math.sin(theta) + (y0 - center[1]) * math.cos(theta) + center[1]
        wall2_lis.append([round(new_x, 3), round(new_y, 3)])
    wall1_ = np.array(wall1_lis)
    wall2_ = np.array(wall2_lis)

    center1 = [np.mean(wall1_[:, 0]), np.mean(wall1_[:, 1])]  # 求楼栋几何中心点坐标
    center2 = [np.mean(wall2_[:, 0]), np.mean(wall2_[:, 1])]
    x_min_1 = np.argmin(wall1_[:, 0])
    x_max_1 = np.argmax(wall1_[:, 0])
    x_min_2 = np.argmin(wall2_[:, 0])
    x_max_2 = np.argmax(wall2_[:, 0])

    length = min(wall1_[x_max_1][0], wall2_[x_max_2][0]) - max(wall1_[x_min_1][0], wall2_[x_min_2][0])
    return length
