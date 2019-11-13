import numpy as np
import matplotlib.pyplot as plt
from geopandas import GeoSeries
from shapely.geometry import Point, Polygon
import random
from es_data import *
from es_utils import ES_FUNC

# 得到房屋体
house_x = [house_wall_raw[i] for i in range(len(house_wall_raw)) if i % 2 == 0]
house_y = [house_wall_raw[i] for i in range(len(house_wall_raw)) if i % 2 == 1]
p = [(i, j) for i, j in zip(house_x, house_y)]
house_wall = Polygon(p)
house_wall = GeoSeries(house_wall)

# 得到红线体
read_line = GeoSeries(Polygon([(i[0], i[1]) for i in read_line_raw]))

# 得到整个红线区域外界矩形的范围
minx = float(read_line.bounds.minx)
maxx = float(read_line.bounds.maxx)
miny = float(read_line.bounds.miny)
maxy = float(read_line.bounds.maxy)

# 各类参数
cross_rate = 0.5 # 交叉率
mutation_rate = 0.2 # 变异率
dna_size = 8  # 一个个体中的点的个数(一个方案中房屋的栋数)
pop_size = 200  # 种群大小
distanch_apart = 50 # 每个房屋之间的距离

if __name__ == '__main__':
    ef = ES_FUNC(house_wall, read_line, cross_rate, mutation_rate, distanch_apart, dna_size, pop_size)

    POP = ef.init_P_total()

    for i in range(10000):
        # ef.select(POP)
        ef.crossover(POP)
        ef.mutate(POP)
        print(i)
        if len(ef.res) == 2:
            print(ef.res)
            break


    res = ef.res
    for x,y in res[0]:
        house_x_tmp = [i + x for i in house_x]
        house_y_tmp = [i + y for i in house_y]
        plt.plot(house_x_tmp,house_y_tmp)

    read_line_x = [i[0] for i in read_line_raw]
    read_line_y = [i[1] for i in read_line_raw]
    plt.plot(read_line_x,read_line_y)
    print(res)
    plt.show()


