import numpy as np
import random

class ES_FUNC:
    def __init__(self, house_wall, read_line, cross_rate,
                 mutation_rate, distanch_apart, dna_size, pop_size):
        """
        house_wall: 房屋线
        read_line: 红线
        cross_rate: 交叉率
        mutation_rate：变异率
        distanch_apart：每个房屋之间的间距
        dna_size: 一个方案中房屋的栋数
        pop_size：一个种群中包含多少个个体
        """
        self.house_wall = house_wall

        self.read_line = read_line
        self.minx = float(self.read_line.bounds.minx)
        self.maxx = float(self.read_line.bounds.maxx)
        self.miny = float(self.read_line.bounds.miny)
        self.maxy = float(self.read_line.bounds.maxy)

        self.cross_rate = cross_rate
        self.mutation_rate = mutation_rate
        self.distanch_apart = distanch_apart
        self.dna_size = dna_size
        self.pop_size = pop_size
        self.res = [] # 存储最大适应度的方案
        self.res2 = []
        # 一个方案的最大适应度(一个方案如果达到了最大适应度，则就是满足条件的方案)
        assert self.dna_size > 1
        # self.max_fitness = (self.dna_size - 1)*2
        self.max_fitness = self.dna_size * (self.dna_size - 1) // 2 + self.dna_size

    def in_area(self, p):
        """
        args:
            p: 房屋坐标，numpy类型，shape为(2, )

        returns:
            房屋在红线内返回1，否则返回0
        """

        h1 = self.house_wall.translate(xoff=p[0], yoff=p[1]) # 将房屋移动到p点处
        aa = h1.within(self.read_line) # 判断移动后的房屋是否在红线内
        return 1 if bool(aa[0]) else 0

    def in_area_total(self, DNA):
        """
        args:
            DNA: 种群中的一个个体，也就是一个排楼方案，numpy类型，shape为(房屋栋数, 2)
        """
        sum = 0
        for i in DNA:
            # if self.in_area(i) == 0: # 如果方案中有点不在区域内，则直接返回0
            #     return 0
            sum += self.in_area(i)
        return sum

    def distance(self, p1, p2):
        h1 = self.house_wall.translate(xoff=p1[0], yoff=p1[1])
        h2 = self.house_wall.translate(xoff=p2[0], yoff=p2[1])
        # 如果某个点不在红线内，则直接返回0(现在已经改成如果方案中有点不在红线内，则直接返回适应度为0，所以这一步不需要了)
        # if self.in_area(p1) == 0 or self.in_area(p2) == 0:
        #     return 0
        d = h1.distance(h2)
        return 1 if float(d[0]) > self.distanch_apart else 0

    def distance_ladder(self, p1, p2):
        h1 = self.house_wall.translate(xoff=p1[0], yoff=p1[1])
        h2 = self.house_wall.translate(xoff=p2[0], yoff=p2[1])
        d = float(h1.distance(h2)[0])
        fitness = 0
        if d > self.distanch_apart:
            fitness = 6
        elif d >= self.distanch_apart * 0.8 and d <= self.distanch_apart:
            fitness = 5
        elif d >= self.distanch_apart * 0.6 and d < self.distanch_apart * 0.8:
            fitness = 4
        elif d >= self.distanch_apart * 0.4 and d < self.distanch_apart * 0.6:
            fitness = 3
        elif d >= self.distanch_apart * 0.2 and d < self.distanch_apart * 0.4:
            fitness = 2
        elif d > 0 and d < self.distanch_apart * 0.2:
            fitness = 1
        else:
            fitness = 0
        return fitness

    def distance_linear(self, p1, p2):
        h1 = self.house_wall.translate(xoff=p1[0], yoff=p1[1])
        h2 = self.house_wall.translate(xoff=p2[0], yoff=p2[1])
        d = h1.distance(h2)
        f = float(d[0]) / self.distanch_apart
        if f >= 1.0:
            fitness = 1
        else:
            fitness = f
        return fitness

    def distance_total(self, DNA):
        l = len(DNA)
        sum = 0
        # 两两之间比对
        for i in range(l - 1):
            for j in range(i + 1, l):
                sum += self.distance_linear(DNA[i], DNA[j])
        return sum

    def distance_total_one_other(self, DNA):
        l = len(DNA)
        sum = 0
        # 两两之间比对
        for i in range(1,l):
            sum += self.distance_linear(DNA[0], DNA[i])
        return sum

    def get_fitness(self, DNA):
        """
        得到一个个体(方案)的适应度
        """
        area_fitness = self.in_area_total(DNA)
        # if area_fitness == 0: # 如果方案中有点不在红线内，则直接这个方案的适应度就为0
        #     return 0
        distance_fitness = self.distance_total(DNA)
        fitness = area_fitness + distance_fitness  # kyt: 适应度只算距离适应度，因为现在方案中只要有不在红线中的点就适应度为0，把是否在区域中做成了一个判断条件了
        # print("distance_fitness=", distance_fitness, "area_fitness=", area_fitness)
        # print("%%%%%%%%%%%%%%%")
        # print(distance_fitness)

        if fitness == self.max_fitness - 0.3:
            flag = True
            # 判断是否为重复的方案
            for n in self.res2:
                if (DNA == n).all():
                    flag = False
                    break
            if flag:
                print(DNA)
                self.res2.append(DNA.copy())  # 注意这里要进行拷贝操作

        if fitness == self.max_fitness:
            flag = True
            # 判断是否为重复的方案
            for n in self.res:
                if (DNA == n).all():
                    flag = False
                    break
            if flag:
                print(DNA)
                self.res.append(DNA.copy()) # 注意这里要进行拷贝操作
        return fitness

    def get_fitness_probability(self, POP):
        """
        POP: 为种群，numpy类型，shape为(种群中方案的个数, 房屋栋数, 2)
        函数返回的依据种群中每个个体的适应度，返回其对应的概率，为选择做准备
        """
        fitness = [self.get_fitness(i) for i in POP]
        # return np.array(fitness) / sum(fitness)
        return np.exp(fitness) / sum(np.exp(fitness)) # 使用softmax放大适应度之间概率的差距
    def get_best(self,POP):
        fitness = [self.get_fitness(i) for i in POP]
        arg = np.argmax(fitness)
        return POP[arg].copy()


    def init_P(self):
        """
        初始化一个个体
        """
        DNA = []
        while True:
            # 将点初始化在红线的中心位置附近
            x = float(self.read_line.centroid.x[0])  + float(np.random.normal(loc=0, scale=100))
            y = float(self.read_line.centroid.y[0])  + float(np.random.normal(loc=0, scale=100))
            if not self.in_area([x,y]):
                continue
            DNA.append([x, y])
            if len(DNA) == self.dna_size:
                break
        return DNA

    def init_P_total(self):
        """
        初始化一个种群
        """
        POP = []
        for i in range(self.pop_size):
            POP.append(self.init_P())
        return np.array(POP)

    def get_best_sort(self, POP):
        fitness = [self.get_fitness(i) for i in POP]
        pop_sorted = np.argsort(fitness)
        return POP[pop_sorted].copy()

    def select(self, POP):
        """
        选择过程，按self.get_fitness_total(POP)返回的概率进行选择，replace表示能够重复选择
        """
        idx = np.random.choice(np.arange(self.pop_size), size=self.pop_size, replace=True, p=self.get_fitness_probability(POP))

        return POP[idx]

    def crossover(self, POP):
        """
        交叉过程，每个个体都可能进行交叉
        会从种群中随机选择一个个体、随机选择个体中的点，把其值赋值给被选择交叉的个体
        """
        first = self.get_best_sort(POP)[-int(self.pop_size * 0.5):]
        for parent in POP:
            if np.random.rand() < self.cross_rate:
                i_ = np.random.randint(0, self.pop_size, size=1)  # 从种群中随机选择一个个体
                cross_points = np.random.randint(0, 2, self.dna_size).astype(np.bool)  # 随机选择个体中的点
                parent[cross_points] = POP[i_, cross_points]
        POP[self.pop_size - int(self.pop_size * 0.5):] = first
        return POP.copy()

    def mutate(self, POP):
        """
        变异过程
        """
        first = self.get_best_sort(POP)[-int(self.pop_size * 0.5):]
        for parent in POP:
            for point in range(self.dna_size):
                x, y = parent[point]
                if np.random.rand() < self.mutation_rate:
                    # x += np.random.normal(2,5)
                    x += random.uniform(-8, 8)
                if np.random.rand() < self.mutation_rate:
                    # y += np.random.normal(2,5)
                    y += random.uniform(-8, 8)

                # 如果点变异到了外接矩形的外面，就将这个数随机到0-其最大值之间
                if x < self.minx or x > self.maxx:
                    x = random.uniform(self.minx, self.maxx)
                if y < self.miny or y > self.maxy:
                    y = random.uniform(self.miny, self.maxy)

                parent[point] = x, y  # choose a random ASCII index
        POP[self.pop_size - int(self.pop_size * 0.5):] = first
        return POP.copy()