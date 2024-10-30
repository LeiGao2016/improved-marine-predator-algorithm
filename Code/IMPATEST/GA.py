import numpy as np
import matplotlib.pyplot as plt


class GA(object):
    def __init__(self, para_array=None, obj_func=None, max_iter=500, mp=0.05, cp=0.9, delta=0.001, population_size=50):
        """
        :param para_array: 参数数组
        :param max_iter: 最大迭代次数
        :param mp: 变异概率
        :param cp: 交叉概率
        :param delta: 放大倍数
        :param population_size:
        """
        self.para_array = para_array
        self.obj_func = obj_func
        self.max_iter = max_iter
        self.mp = mp
        self.cp = cp
        self.delta = delta
        self.population_size = population_size

    # 根据解的精度确定染色体(chromosome)的长度——确定二进制编码的长度
    # 需要根据决策变量的上下边界来确定
    # 需要根据决策变量的上下边界来确定
    def get_encoded_length(self):
        """
        :return:  每一个超参的二进制表示的长度
        """
        # 每个变量的编码长度
        lengths = []
        # 根据给定的定义域（自变量的范围）来获取其二进制的长度
        for i in self.para_array:
            lower = i[0]
            upper = i[1]
            while ((np.log2((upper - lower) / self.delta)) + 1) < 6:
                self.delta = self.delta / 10
            # print((upper - lower) / delta)
            # res = fsolve(lambda x: ((upper - lower) * 1 / delta) - 2 ** x + 1, np.array([50]))
            # length = int(np.ceil(res[0]))
            # 得到十进制数的二进制的位数
            length = int(np.log2((upper - lower) / self.delta)) + 1
            # print(length)
            lengths.append(length)
        return lengths

    # GA NI_GA  PSO  模拟退火

    # 随机生成初始编码种群——二进制编码    (编码长度，种群大小)
    # 第一代
    def get_initial_population(self, encode_length, population_size):
        """
        :param encode_length: 二进制编码的长度
        :param population_size: 种群的大小
        :return: 初始种群
        """
        # 种群初始化
        chromosomes = np.zeros((population_size, sum(encode_length)), dtype=np.uint8)
        # 以0，1随机生成初代种群（染色体/个体的集合）
        for i in range(population_size):
            chromosomes[i, :] = np.random.randint(0, 2, sum(encode_length))  # 随机生成 encode_length个0和1
        return chromosomes

    # 染色体解码得到表现型的解，染色体=个体，每个0，1代表基因
    def decoded_chromosome(self, encode_length, chromosomes, boundary_list):
        """
        :param encode_length: 多个种群组成的矩阵
        :param chromosomes: 编码后的种群
        :param boundary_list: 含有多个超参数二进制长度的列表
        :return: 解码后的种群
        """
        populations = chromosomes.shape[0]  # 染色体的个数 10
        variables = len(encode_length)  # 染色体片段数（种群的个数） 2
        decoded_values = np.zeros((populations, variables))  # 解码出来的个体的存储ndarray
        for k, chromosome in enumerate(chromosomes):
            # print(k, chromosome)
            chromosome = chromosome.tolist()  # 将每个单独的染色体从ndarray转换为列表
            start = 0
            for index, length in enumerate(encode_length):
                # 将一个染色体进行拆分，得到染色体片段，也就相当于将每个二进制组拆分，得到每个变量的二进制片段
                power = length - 1
                # 将二进制转换为十进制
                decimal = 0
                for i in range(start, start + length):
                    decimal += chromosome[i] * (2 ** power)
                    power -= 1

                # 得到初始的十进制数，再次存为lower，upper
                lower = boundary_list[index][0]
                upper = boundary_list[index][1]
                # 将前面二进制和转换出的十进制缩小倍数再加上lower，得到在Lower和upper中间的十进制数
                decoded_value = lower + decimal * (upper - lower) / (2 ** length - 1)
                # (upper - lower) / (2 ** length - 1) 表示精度
                decoded_values[k, index] = decoded_value
                # 将start也就是二进制的索引位置移到染色体下一段的开始位置
                start = length
        return decoded_values
        # 得到一矩阵，最外围维度表示种群中个体数量，下一维度表示决策变量（染色体片段数或种群个数）个数

    def get_fitness_value(self, func, chromosomes_decoded):
        """
        :param func: 求最优解的函数
        :param chromosomes_decoded: 解码后的种群集合
        :return:
                fitness_values: 染色体片段（个体）的适应度值
                cum_probability: 每个个体被选择的累积概率
        """
        # 得到种群的染色体数（个体数）population，染色体片段数或种群数（决策变量的个数）nums
        population, nums = chromosomes_decoded.shape
        # 初始化种群的适应度值为0
        fitness_values = np.zeros((population, 1))
        # 计算适应度值,其实就是所求的函数值，因为根据函数的大小判断其是否保留，所以也叫适应度值
        # print(type(chromosomes_decoded[0, :]))
        for i in range(population):
            fitness_values[i, 0] = func(self.obj_func, chromosomes_decoded[i, :])
        # print(fitness_values)
        # 轮盘赌选择法，
        # 计算每个染色体被选择的概率
        divide_num = np.sum(fitness_values)
        if divide_num == 0:  # cg
            divide_num = 1
        probability = fitness_values / divide_num
        # print(probability)
        # print(probability)
        # 得到前n个染色体被选中的概率
        cum_probability = np.cumsum(probability)
        # print(cum_probability)
        return fitness_values, cum_probability

    def select_new_population(self, chromosomes, cum_probability):
        """
        :param chromosomes: 上一代种群
        :param cum_probability: 累计概率
        :return: 新一代种群
        """
        # 上一代种群的个数m.染色体片段数（或个体数）n
        m, n = chromosomes.shape
        # 新一代的种群
        new_population = np.zeros((m, n), dtype=np.uint8)
        # 随机产生m个概率值x,用于与初代累计概率进行比较
        randoms = np.random.rand(m)
        # 为保障种群的染色体数目（个体数）不变，将生成的m个随机数与上一代种群比较，再次生成新的含有m个个体的种群
        for i, random_a in enumerate(randoms):
            logical = cum_probability >= random_a
            # print(logical)
            # 当r<cum_[1]，则选择个体1，
            # 否则，选择个体k，使得：q[k-1]<r≤q[k] 成立；
            index = np.where(logical == 1)
            # 将第一个满足条件的染色体（或个体）放入新一代种群，可能会有相同个体放入多次。index是tuple,tuple中元素是ndarray
            if index[0].shape[0] == 0:  # cg
                continue
            new_population[i, :] = chromosomes[index[0][0], :]
            # 在累计概率中最后一个元素值为1，可保证index[0][0]永远不为空
        # print(new_population)
        # 返回新一代种群（二进制形式）
        return new_population

    def crossover(self, population):
        """
        种群交叉
        Pc: 交叉概率默认是0.8
        :param population: 新种群
        :return: 交叉后得到的新种群
        """
        # 种群的染色体数（个体数）m，每个染色体的基因数量n
        m, n = population.shape
        # 使用uint8存储，顺便取整
        numbers = np.uint8(m * self.cp)
        # 确保进行交叉的染色体个数是偶数个
        if numbers % 2 != 0:
            numbers += 1
        # 交叉后得到的新种群
        update_population = np.zeros((m, n), dtype=np.uint8)
        # 在所选择出的新种群中进行交叉，交叉过后，替换掉原来的。
        # 根据原始种群的染色体总数（个体总数）产生随机索引，比如总数为10，则从集合[0,9]中不重复抽取numbers个数字
        index = np.random.choice(range(m), numbers, replace=False).tolist()
        # 从序列range(m)中随机截取number长的片段
        # 不进行交叉的染色体进行复制（不在index里面的种群直接传给update_population）
        for i in range(m):
            # if not index.__contains__(i):
            if i not in index:
                update_population[i, :] = population[i, :]
        # crossover
        while len(index) > 0:
            # 从后向前依次选取，直至没有可以进行交叉的个体
            a = index.pop()  # a 表示index中最后一个元素值
            b = index.pop()  # b 表示index中倒数第二个元素值
            # 在一个染色体的基因序列中随机选取一个交叉点
            crossover_point = np.random.choice(range(1, n), 1, replace=False)[0]
            # 在染色体中确认交叉点的位置,也就是索引为a、b的染色体在选择的交叉点进行变换

            update_population[a, 0:crossover_point] = population[a, 0:crossover_point]
            update_population[a, crossover_point:] = population[b, crossover_point:]
            update_population[b, 0:crossover_point] = population[b, 0:crossover_point]
            update_population[b, crossover_point:] = population[a, crossover_point:]
        # 返回交叉后的种群（二进制）
        return update_population

    def mutation(self, population):
        """
        种群染色体变异
        Pm: 变异概率默认是0.01
        :param population: 经交叉后得到的种群
        :return: 经变异操作后的新种群
        """
        update_population = np.copy(population)
        # 种群的染色体数（个体数）m，每个染色体的基因数量n
        m, n = population.shape
        # 计算需要变异的基因个数
        gene_num = np.uint8(m * n * self.mp)
        # 将所有的基因按照序号进行10进制编码，则共有m*n个基因
        # 随机抽取gene_num个基因进行基本位变异
        mutation_gene_index = np.random.choice(range(0, m * n), gene_num, replace=False)
        # 确定每个将要变异的基因在整个染色体中的基因座(即基因的具体位置)
        for gene in mutation_gene_index:
            # 确定变异基因位于第几个染色体
            chromosome_index = gene // n
            # 确定变异基因位于当前染色体的第几个基因位
            gene_index = gene % n
            # 变异，1变成0，0变成1
            if update_population[chromosome_index, gene_index] == 0:
                update_population[chromosome_index, gene_index] = 1
            else:
                update_population[chromosome_index, gene_index] = 0
        # 变异后的种群
        return update_population

    def fitnessFunction(self, obj_func, para_list):
        """
           定义适应度函数
               返回：
               return: 适应度函数的值
        """
        # return lambda x: 21.5 + x[0] * np.sin(4 * np.pi * x[0]) + x[1] * np.sin(20 * np.pi * x[1])
        # if name == 'gaussian':
        #     return lambda p, theta: np.exp(-theta * np.abs(x[0] - x[1]) ** p / (2 * x[3] ** 2))
        # return lambda x: x[0] ** 2
        # return np.sum(para_list ** 2)
        return obj_func(para_list)

    def genetic_algorithm(self):
        # 每次迭代得到的最优解
        optimal_solutions = []
        optimal_values = []
        # 决策变量（自变量）的取值范围，维数多就增加数组元素个数，例如[[4.1, 5.8]，[8, 10]]
        #
        # decision_variables = [[100, 100.5], [4.1, 5.8]]
        # 得到染色体编码长度
        length_encode = self.get_encoded_length()
        # print(length_encode)
        # 得到初始种群编码
        chromosomes_encoded = self.get_initial_population(length_encode, self.population_size)
        # 种群解码
        decoded = self.decoded_chromosome(length_encode, chromosomes_encoded, self.para_array)
        # evalvalues, cum_proba = get_fitness_value(fitnessFunction(), decoded)
        # 得到个体适应度值和个体的累积概率
        fitness_values, cum_individual_proba = self.get_fitness_value(self.fitnessFunction, decoded)
        for iteration in range(self.max_iter):
            # 选择新的种群
            new_populations = self.select_new_population(chromosomes_encoded, cum_individual_proba)
            # 进行交叉操作
            crossover_population = self.crossover(new_populations)
            # mutation进行变异
            mutation_population = self.mutation(crossover_population)
            # 将变异后的种群解码，得到每轮迭代最终的种群
            final_decoded = self.decoded_chromosome(length_encode, mutation_population, self.para_array)
            # 变异后的适应度值，累计概率
            fitness_values, cum_individual_proba = self.get_fitness_value(self.fitnessFunction, final_decoded)
            # 搜索每次迭代的最优解（当前是取最小值，所以是min），以及最优解对应的目标函数（十进制染色体）的取值
            # optimal_values.append(np.max(list(fitness_values)))
            optimal_values.append(np.min(list(fitness_values)))
            # index = np.where(fitness_values == min(list(fitness_values)))
            # optimal_solutions.append(final_decoded[index[0][0], :])
        # 从每次迭代的最优解集合中找到总体的最优解
        optimal_value = np.min(optimal_values)
        return optimal_value, optimal_values


# 下面是6个函数，其中para_list是搜索空间，也就是自变量
def Sphere(para_list):
    return np.sum(para_list ** 2)


# 对这6个函数使用GA测试的入口，分别可以测试空间dv
def test_func(dv, obj_func, mp, cp, max_iter, pop_size):
    """
    :param dv:  测试空间，也就是自变量
    :param obj_func: 测试函数
    :param mp: 突变概率
    :param cp: 交叉概率
    :param max_iter: 最大迭代次数
    :param pop_size: 种群数量
    :return: 最优值
    """
    ga = GA(para_array=dv, obj_func=obj_func, mp=mp, cp=cp, max_iter=max_iter, population_size=pop_size)
    value, MM = ga.genetic_algorithm()
    return value, MM

decision_variables_100 = [[-100, 100]] * 12


# 对每一个函数进行具体测试，可以改变的是以下四个参数变量
def test_all_func(mp, cp, max_iter, pop_size):
    value_sphere,MM = test_func(decision_variables_100, Sphere, mp, cp, max_iter, pop_size)
    return value_sphere, MM


max_iter = 500
value_sphere,MM = test_all_func(0.05, 0.9, max_iter, 50)
x_Max_iteration = np.arange(0, max_iter, 1)
x_Max_iteration = x_Max_iteration.reshape(1, max_iter)
MM = np.array(MM)
plt.plot(x_Max_iteration[0], MM, linestyle="-.", color='green', lw=2)
plt.legend(['GA'], loc='lower left')
plt.title('F1 Convergence_curve')
plt.xlabel('Iter')
plt.ylabel('Convergence value')
plt.yscale('log')
plt.show()
print(MM.shape)