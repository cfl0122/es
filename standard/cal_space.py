# -*- coding: utf-8 -*-
# import matplotlib.pyplot as plt
import numpy as np
from sympy import *
import math
import numpy as np
import time
import geopandas as gpd
from shapely.geometry import Polygon

class ValError(ValueError):
    pass


class CalSpace:
    def __init__(self, wall1, wall2, center, angle):
        self.wall1 = wall1  # 建筑一的轮廓
        self.wall2 = wall2  # 建筑二的轮廓
        self.rotate_center = center  # 旋转中心
        self.angle = angle  # 旋转角度
        self.min_dist = self.cal_space()

    # 判定楼栋上某一点向某方向投射的时候是否穿过自身
    def is_inside_polygon(self, pt, poly, north):
        ind = 999
        try:
            ind = poly.index(pt)
        except:
            pass
        i = -1
        j = len(poly) - 1
        if north:  # 研究的楼栋位于北侧时，采用下射线
            while i < len(poly) - 1:
                i += 1
                # 采用射线法判断点的下射线和边线的交点个数，为奇数则在内部，为偶数则为外部。
                if poly[i][0] <= pt[0] < poly[j][0] or poly[j][0] <= pt[0] < poly[i][0]:
                    mark = round(
                        (poly[j][1] - poly[i][1]) * (pt[0] - poly[i][0]) / (poly[j][0] - poly[i][0]) + poly[i][1], 3)
                    if round(pt[1], 3) > mark:
                        if ind != 999 and ind in [i - 1, i + 1, j - 1, j + 1] and pt[0] in [poly[i][0], poly[j][0]]:
                            return False
                        return True
                j = i
            return False
        else:  # 研究的楼栋位于南侧时，采用上射线
            while i < len(poly) - 1:
                i += 1
                # 采用射线法判断点的上射线和边线的交点个数，为奇数则在内部，为偶数则为外部。
                if poly[i][0] <= pt[0] < poly[j][0] or poly[j][0] <= pt[0] < poly[i][0]:
                    mark = round(
                        (poly[j][1] - poly[i][1]) * (pt[0] - poly[i][0]) / (poly[j][0] - poly[i][0]) + poly[i][1], 3)
                    if round(pt[1], 3) < mark:
                        # 排除纵向连续共线的情况
                        if ind != 999 and ind in [i - 1, i + 1, j - 1, j + 1] and pt[0] in [poly[i][0], poly[j][0]]:
                            return False
                        return True
                j = i
            return False

    # 计算pt在line上的投影点
    def check_on_line(self, pt, line):  # x_extreme 为另一楼栋极左边和极右边的X坐标
        [x1, y1], [x2, y2] = line
        k = (y2 - y1) / (x2 - x1)
        b = y1 - k * x1
        y = k * pt[0] + b
        return [pt[0], y]

    # 轮廓挑选初始化
    def initial(self, center1, center2, wall):
        # 被初始的轮廓center2 和wall， 另一栋楼center1
        north = False
        if center2[1] > center1[1]:
            north = True
        num = list()  # 记录直接面对另一楼栋的点的编号
        for i in range(len(wall)):
            if not self.is_inside_polygon(wall[i], wall, north):
                num.append(i)
        # 计算本楼栋投射到自己身上的阴影处的临界点坐标，并添加一系列num的轮廓点中
        res = list()
        for j in range(len(num)):
            if num[j] - 1 not in num:  # 寻找右边是否有临界点
                for x in num:
                    if wall[num[j] - 1][0] < wall[x][0] < wall[num[j]][0] or wall[num[j]][0] < wall[x][0] < \
                            wall[num[j] - 1][0]:
                        pt = self.check_on_line(wall[x], [wall[num[j] - 1], wall[num[j]]])
                        if not self.is_inside_polygon(pt, wall, north):
                            res.append(pt)
            res.append(wall[num[j]])
            if num[j] + 1 not in num and num[j] + 1 < len(wall):  # 寻找左边是否有临界点
                for x in num:
                    if wall[num[j] + 1][0] < wall[x][0] < wall[num[j]][0] or wall[num[j]][0] < wall[x][0] < \
                            wall[num[j] + 1][0]:
                        pt = self.check_on_line(wall[x], [wall[num[j] + 1], wall[num[j]]])
                        if not self.is_inside_polygon(pt, wall, north):
                            res.append(pt)
            elif num[j] + 1 not in num and num[j] + 1 >= len(wall):  # 寻找左边是否有临界点，且编号跨过了0
                for x in num:
                    if wall[0][0] < wall[x][0] < wall[num[j]][0] or wall[num[j]][0] < wall[x][0] < wall[0][0]:
                        pt = self.check_on_line(wall[x], [wall[0], wall[num[j]]])
                        if not self.is_inside_polygon(pt, wall, north):
                            res.append(pt)
        # 将轮廓从左到右重新排序
        mark = np.argmin(np.array(res)[:, 0])
        temp = res[mark:] + res[:mark]
        return temp

    # 截取楼栋wall被extreme（东西方向上）重叠遮挡的部分
    def get_loop(self, extreme, wall):
        cover = list()
        for i in range(len(wall) - 1):
            if extreme[0] <= wall[i][0] <= extreme[1]:
                cover.append(list(wall[i]))
            elif extreme[0] < wall[i + 1][0] < extreme[1]:  # 当某线段横跨极点，则求垂点，截取新线段
                cover.append(self.critical_line([wall[i], wall[i + 1]], [extreme[0], extreme[1]]))
            elif extreme[0] < wall[i - 1][0] < extreme[1] and i > 0:  # 当某线段横跨极点，则求垂点，截取新线段
                cover.append(self.critical_line([wall[i - 1], wall[i]], [extreme[0], extreme[1]]))
            # 当某线段覆盖了极点在x轴上的投影
            elif (wall[i][0] < extreme[0] < extreme[1] < wall[i + 1][0]) or (
                    wall[i + 1][0] < extreme[0] < extreme[1] < wall[i][0]):
                cover += (self.critical_line([wall[i], wall[i + 1]], [extreme[0], extreme[1]]))
        if extreme[0] <= wall[-1][0] <= extreme[1]:
            cover.append(list(wall[-1]))
        return cover

    # 求线段被另一楼栋遮到的临界点坐标
    def critical_line(self, line, x_extreme):  # x_extreme 为另一楼栋极左边和极右边的X坐标
        [x1, y1], [x2, y2] = line
        if x1 < x_extreme[0] < x_extreme[1] < x2 or x2 < x_extreme[0] < x_extreme[1] < x1:  # 极点在线内的情况
            try:
                y_min = (y2 - y1) * (x_extreme[0] - x1) / (x2 - x1) + y1
                y_max = (y2 - y1) * (x_extreme[1] - x1) / (x2 - x1) + y1
            except:
                raise ValError("The inputs are wrong.")
            return [[x_extreme[0], y_min], [x_extreme[1], y_max]]
        elif x1 < x_extreme[0] < x2 or x2 < x_extreme[0] < x1:  # 右边极点和线相交的情况
            try:
                y = (y2 - y1) * (x_extreme[0] - x1) / (x2 - x1) + y1
            except:
                raise ValError("The inputs are wrong.")
            return [x_extreme[0], y]
        elif x1 < x_extreme[1] < x2 or x2 < x_extreme[1] < x1:  # 左边极点和线相交的情况
            try:
                y = (y2 - y1) * (x_extreme[1] - x1) / (x2 - x1) + y1
            except:
                raise ValError("The inputs are wrong.")
            return [x_extreme[1], y]

    # 计算点到线段的距离，垂点如果在线段上，则保留结果，否则返回None
    def point_line_dist(self, line, point):
        [x1, y1], [x2, y2] = line
        x, y = symbols('x y')
        res = (solve(
            [(y2 - y) * (x2 - x1) - (y2 - y1) * (x2 - x), (y - point[1]) * (y1 - y2) + (x - point[0]) * (x1 - x2)]))
        if x1 < res[x] < x2 or x2 < res[x] < x1:
            return [res[x], res[y]]
        else:
            return None

    # 计算两楼栋轮廓之间最短距离
    def cal_space(self):
        wall1 = np.reshape(self.wall1, (-1, 2))
        wall2 = np.reshape(self.wall2, (-1, 2))
        theta = np.pi * self.angle / 180  # 以顺时针旋转为正
        center = self.rotate_center
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
        extreme1 = [wall1_[x_min_1][0], wall1_[x_max_1][0]]  # 求楼栋最左最右的x坐标
        extreme2 = [wall2_[x_min_2][0], wall2_[x_max_2][0]]
        """仅用于判定是否存在遮挡现象"""
        if extreme1[0] > extreme2[1] or extreme1[1] < extreme2[0]:
            return False

        filter1 = self.initial(center2, center1, wall1_lis)
        filter2 = self.initial(center1, center2, wall2_lis)
        cover2 = self.get_loop(extreme1, filter2)
        cover1 = self.get_loop(extreme2, filter1)
        while len(cover1) < 3:
            cover1.append([cover1[-1][0] + 0.1, cover1[-1][1] + 0.1])
        while len(cover2) < 3:
            cover2.append([cover2[-1][0] + 0.1, cover2[-1][1] + 0.1])   # 计算最短距离
        w1 = np.reshape(cover1, (-1, 2))
        w2 = np.reshape(cover2, (-1, 2))

        w1 = Polygon(w1)
        w1 = gpd.geoseries.GeoSeries(w1)

        w2 = Polygon(w2)
        w2 = gpd.geoseries.GeoSeries(w2)

        dist = w2.distance(w1)

        return dist


if __name__ == "__main__":
    wall1 = [65.09999917132927, 103.07499885787331, 58.59999917152982, 103.07499885787331, 58.59999917152982,
             103.6749988622247, 56.69999917139239, 103.67499886223766, 56.6999929935414, 103.07492416496837,
             53.89999917147361, 103.07499886221956, 53.89999861724982, 105.77499886155476, 51.59999917158393,
             105.77499886160388, 51.59999917158393, 106.82499886163089, 48.49999944837191, 106.82498775555462,
             47.49999917129435, 106.82500113772005, 47.49999917113655, 105.42500341302241, 46.349999171406125,
             105.42499885782934, 46.34999917153482, 103.07500158432651, 43.29999271700808, 103.07499886224434,
             43.300005626182156, 103.67498903254443, 41.39999917109117, 103.6749988622247, 41.39999917109117,
             103.07499885787331, 34.89999917153909, 103.07500868757336, 34.89999917153909, 94.97499885858872,
             65.09999971993057, 94.97499885858872]
    wall2 = [65.91186108993563, 40.22358528485271, 31.331716717052963, 49.48930704371825, 33.86814325602846,
             58.95537975684089, 45.12117930454367, 55.94013783524292, 45.988222618661794, 59.17598905861163,
             50.8178518576467, 57.88189422952602, 50.63670826584334, 57.205857143230226, 53.05152282897179,
             56.55880953116905, 53.232666314008775, 57.234846221092894, 58.06229544697007, 55.940750995002254,
             57.19525137145095, 52.70489997480386, 68.44828762891112, 49.68965799797535]
    # 旋转楼栋令要求的遮挡射线沿着南北方向
    center = [10, 0]  # 取某楼栋的中心点
    angle = 30

    res = CalSpace(wall1, wall2, center, angle).min_dist
    print(res)
