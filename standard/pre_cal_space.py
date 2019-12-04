# -*- coding: utf-8 -*-
# from cal_space import CalSpace  # 精确，但时间增加十倍
import numpy as np
import math
from standard.cal_space import CalSpace  # 粗略，快


class PreCalSpace:
    def pre_cal_space(self, wall1, wall2, angle, center, sign="平行"):
        # wall1, wall2 为楼栋轮廓坐标点的一维list，angle为旋转角度，center为旋转中心，sign表示检测消防最短间距或是普通的方位间距
        origin = [0, 0, 0, -1]  # 角度基准向量为正南
        if sign == "最短":
            wall1_ = np.reshape(wall1, (-1, 2))
            wall2_ = np.reshape(wall2, (-1, 2))
            center1 = [np.mean(wall1_[:, 0]), np.mean(wall1_[:, 1])]  # 求楼栋几何中心点坐标
            center2 = [np.mean(wall2_[:, 0]), np.mean(wall2_[:, 1])]
            included_angle = self.angle(origin, center1 + center2)
            res = CalSpace(wall1, wall2, center, included_angle).min_dist
        else:
            res = CalSpace(wall1, wall2, center, angle).min_dist
        try:
            res2 = res[0]
        except:
            res2 = False
        return res2

    def angle(self, v1, v2):
        dx1 = v1[2] - v1[0]
        dy1 = v1[3] - v1[1]
        dx2 = v2[2] - v2[0]
        dy2 = v2[3] - v2[1]
        angle1 = math.atan2(dy1, dx1)
        angle1 = int(angle1 * 180 / math.pi)
        # print(angle1)
        angle2 = math.atan2(dy2, dx2)
        angle2 = int(angle2 * 180 / math.pi)
        # print(angle2)
        if angle1 * angle2 >= 0:
            included_angle = abs(angle1 - angle2)
        else:
            included_angle = abs(angle1) + abs(angle2)
            if included_angle > 180:
                included_angle = 360 - included_angle
        return included_angle
        

if __name__ == "__main__":
    # t0 = time.time()
    wall1 = [65.09999917132927, 103.07499885787331, 58.59999917152982, 103.07499885787331, 58.59999917152982, 103.6749988622247, 56.69999917139239, 103.67499886223766, 56.6999929935414, 103.07492416496837, 53.89999917147361, 103.07499886221956, 53.89999861724982, 105.77499886155476, 51.59999917158393, 105.77499886160388, 51.59999917158393, 106.82499886163089, 48.49999944837191, 106.82498775555462, 47.49999917129435, 106.82500113772005, 47.49999917113655, 105.42500341302241, 46.349999171406125, 105.42499885782934, 46.34999917153482, 103.07500158432651, 43.29999271700808, 103.07499886224434, 43.300005626182156, 103.67498903254443, 41.39999917109117, 103.6749988622247, 41.39999917109117, 103.07499885787331, 34.89999917153909, 103.07500868757336, 34.89999917153909, 94.97499885858872, 65.09999971993057, 94.97499885858872]
    wall2 = [65.91186108993563, 40.22358528485271, 31.331716717052963, 49.48930704371825, 33.86814325602846, 58.95537975684089, 45.12117930454367, 55.94013783524292, 45.988222618661794, 59.17598905861163, 50.8178518576467, 57.88189422952602, 50.63670826584334, 57.205857143230226, 53.05152282897179, 56.55880953116905, 53.232666314008775, 57.234846221092894, 58.06229544697007, 55.940750995002254, 57.19525137145095, 52.70489997480386, 68.44828762891112, 49.68965799797535]
    # 旋转楼栋令要求的遮挡射线沿着南北方向
    center = [10, 0]  # 取某楼栋的中心点
    angle = -10
    sign = "最短"
    res = PreCalSpace().pre_cal_space(wall1, wall2, angle, center, sign)
    print(res)
    # t1 = time.time()
    # print("计算一次时间：", t1 - t0)

