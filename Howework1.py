import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import random
import sys


class Point:

    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __str__(self):
        result = "point("
        for i in range(len(self.data)):
            if (i + 1) == len(self.data):
                result = result + str(self.data[i]) + ")"
            else:
                result = result + str(self.data[i]) + ","
        return result

    def translate(self, dimension, distance):
        if distance <= 0:
            raise ValueError('must be positive')
        if dimension < 0 or dimension >= len(self.data):
            raise ValueError('invalid dimension')
        new_data = self.data
        new_data[dimension] = self.data[dimension] + distance
        return Point(new_data)

    def translateByVector(self, vector):
        new_data = self.data
        for i in range(len(self.data)):
            new_data[i] = self.data[i] + vector.data[i]
        return Point(new_data)

    # 相對原點的向量
    def vector(self):
        return Vector(self.data)


class Vector:
    def __init__(self, data):
        self.data = []
        for i in range(len(data)):
            self.data.append(data[i])

    def __eq__(self, obj):
        if obj is None:
            return False
        for i in range(len(self.data)):
            if self.data[i] != obj.data[i]:
                return False
        return True

    def __str__(self):
        result = "vector("
        for i in range(len(self.data)):
            if (i + 1) == len(self.data):
                result = result + str(self.data[i]) + ")"
            else:
                result = result + str(self.data[i]) + ","
        return result

    def length(self):
        total = 0
        for i in range(len(self.data)):
            total += np.square(self.data[i])
        return np.sqrt(total)

    def dotProduct(self, otherVector):
        total = 0
        for i in range(len(self.data)):
            total = total + (self.data[i] * otherVector.data[i])
        return total

    def scale(self, factor):
        new_data = self.data
        for i in range(len(self.data)):
            new_data[i] = self.data[i] * factor
        return Vector(new_data)

    def normalization(self):
        return self.scale(1/self.length())

    def copy(self):
        new_data = self.data
        return Vector(new_data)

    # 向量加法
    def addition(self, vector):
        new_data = self.data
        for i in range(len(self.data)):
            new_data[i] = self.data[i] + vector.data[i]
        return Vector(new_data)

    @staticmethod
    def vectorBetween(fromPoint, toPoint):
        if len(fromPoint.data) != len(toPoint.data):
            raise ValueError('invalid dimension')
        new_data = toPoint.data
        for i in range(len(fromPoint.data)):
            new_data[i] = new_data[i] - fromPoint.data[i]
        return Vector(new_data)


class MLSchool:
    min = -100
    max = 100
    num_points = 100
    precision = 0.01
    dimension = 4
    # pocket algo
    choosePoint1 = -1
    choosePoint2 = -1
    target_function_points_num = -1
    # pocket algo end
    points = []
    test_points = []
    group1_points = []
    group2_points = []
    group1_points_x = []
    group2_points_x = []
    group1_points_y = []
    group2_points_y = []
    target_function_vector = None
    pre_target_function_vector = None
    multi_factor = 0.5
    target_function_const = 1
    tmp_function_const = 0
    calculate_times = 0
    sign = lambda a: -1.0 if a <= 0 else 1.0

    def init_data(self):
        # 初始化數據
        all_data = pd.read_table('hw.dat', header=None, sep='\s+')
        for index, row in all_data.iterrows():
            sub = row[:4].values
            self.points.append(Point(sub, row[4]))

    def init_pa_data(self):
        # 初始化數據
        all_data = pd.read_table('pa_train.dat', header=None, sep='\s+')
        for index, row in all_data.iterrows():
            sub = row[:4].values
            self.points.append(Point(sub, row[4]))

    def init_pa_test_data(self):
        # 初始化數據
        all_data = pd.read_table('pa_test.dat', header=None, sep='\s+')
        for index, row in all_data.iterrows():
            sub = row[:4].values
            self.test_points.append(Point(sub, row[4]))

    @staticmethod
    def get_function_number(point, vector=None, const=None):
        if vector is None:
            vector = MLSchool.target_function_vector
        if const is None:
            const = MLSchool.target_function_const
        result = const
        # print(str(vector)+","+str(point))
        for i in range(MLSchool.dimension):
            result = result + vector.data[i] * point.data[i]
        return result

    def check_sign_fail(self, point, predict_sign):
        number = MLSchool.get_function_number(point)
        if MLSchool.sign(number) != predict_sign:
            # 嘗試更換常數項, 更換後更新常數並重繪製
            # for i in np.arange(self.min, self.max, self.precision):
            #     self.tmp_function_const = i
            #     number = MLSchool.get_function_number(point, None,  self.tmp_function_const)
            #     if MLSchool.sign(number) == predict_sign:
            #         MLSchool.target_function_const = self.tmp_function_const
            #         break
            # 更新向量
            MLSchool.pre_target_function_vector = MLSchool.target_function_vector.copy()
            MLSchool.target_function_vector = MLSchool.target_function_vector \
                .addition(point.vector()).scale(0.5).normalization()
            return True
        return False

    def perceptron_learning_algorithm(self, seed):
        if MLSchool.target_function_vector is None:
            print(seed)
            print(len(self.points))
            MLSchool.target_function_vector = self.points[seed].vector().normalization()
        MLSchool.calculate_times += 1
        print('pla execute - '+str(MLSchool.calculate_times) + "," + str(MLSchool.target_function_vector) +
              ',' + str(MLSchool.pre_target_function_vector) + ',' + str(MLSchool.target_function_vector == MLSchool.pre_target_function_vector))
        for i in range(len(self.points)):
            if MLSchool.target_function_vector != MLSchool.pre_target_function_vector \
                    and self.check_sign_fail(self.points[i], self.points[i].label and MLSchool.calculate_times <= 50):
                self.perceptron_learning_algorithm(seed)
                break

    def next_main_point(self, seed):
        self.choosePoint1 = seed

    def next_end_point(self, seed):
        self.choosePoint2 = seed

    def get_vector(self):
        p1 = self.points[self.choosePoint1]
        p2 = self.points[self.choosePoint2]
        return Vector.vectorBetween(p1, p2)

    def check_all_points(self, vector):
        result = -1
        # 找最少錯誤點
        for j in range(len(self.points)):
            number = MLSchool.get_function_number(self.points[j], vector, None)
            # print("("+str(self.points[j].data[0])+","+str(self.points[j].data[1])+") - " + "," +str(number) +
            #       ","+str(group_no)+","+str(self.group_sign[group_no])+","+str(MLSchool.sign(number)))
            if MLSchool.sign(number) != self.points[j].label:
                result = result + 1
        return result

    def valid_test_points(self):
        result = 0
        for j in range(len(self.test_points)):
            number = MLSchool.get_function_number(self.test_points[j])
            if MLSchool.sign(number) != self.points[j].label:
                result = result + 1
        return result

    def pocket_algorithm(self, seed):
        times_limit = 100
        # 有下個起點且錯誤數量不為零則繼續找
        while times_limit > 0 and MLSchool.target_function_points_num != 0:
            # print('pocket_algorithm execute - '+str(times_limit) + "," +
            #       str(MLSchool.target_function_vector) + ',' + str(MLSchool.target_function_const))
            times_limit = times_limit - 1
            self.next_main_point(seed)
            # 有下個終點且錯誤數量不為零則繼續找
            seed2 = random.randint(0, len(self.points) - 1)
            while seed2 == seed:
                seed2 = random.randint(0, len(self.points) - 1)
            self.next_end_point(seed2)
            vector = self.get_vector().normalization()
            if MLSchool.target_function_vector is None:
                MLSchool.target_function_vector = vector
            wrongs = self.check_all_points(vector)
            if wrongs < MLSchool.target_function_points_num or MLSchool.target_function_points_num == -1:
                MLSchool.target_function_vector = vector
                MLSchool.target_function_points_num = wrongs
                # MLSchool.target_function_const = self.tmp_function_const
            seed = random.randint(0, len(self.points) - 1)

    def check(self, group):
        cou = 0
        for i in range(len(group)):
            number = MLSchool.get_function_number(group[i])
            if number > 0:
                cou += 1
            no = MLSchool.get_group_no_by_group_condition(group[i])
            print("check ("+str(group[i].data[0])+","+str(group[i].data[1])+") - " +str(number)+","+str(self.group_sign[no]))
        print(cou)

sys.setrecursionlimit(1000000)
school = MLSchool()
# school.init_data()

# school.pocket_algorithm()
# school.check(school.group1_points)
# print("XXXXXXX")
# school.check(school.group2_points)
random.seed('magic')
# all_times = 0
# for i in range(2000):
#     seed = random.randint(0, len(school.points) - 1)
#     MLSchool.target_function_vector = None
#     school.perceptron_learning_algorithm(seed)
#     all_times += MLSchool.calculate_times
#     MLSchool.calculate_times = 0
#     print(str(i)+","+str(seed) +","+str(all_times))


school.init_pa_data()
school.init_pa_test_data()
all_wrongs = 0
for i in range(2000):
    seed = random.randint(0, len(school.points) - 1)
    MLSchool.target_function_vector = None
    MLSchool.target_function_points_num = -1
    MLSchool.calculate_times = 0
    # school.pocket_algorithm(seed)
    school.perceptron_learning_algorithm(seed)
    wrongs = school.valid_test_points()
    wrong_rate = wrongs / len(school.test_points)
    all_wrongs = all_wrongs + wrong_rate
    print(str(i)+","+str(seed) + "," +str(wrongs) +","+ str(wrong_rate))

print("done - avarage wrong rates:"+str(all_wrongs/2000))
# print("done - execute times:"+str(MLSchool.calculate_times))
# print("done - function_const:" + str(MLSchool.target_function_const)+","+str(MLSchool.target_function_points_num))
# print("done - target_function_vector:" + str(MLSchool.target_function_vector.data[0]) + "," + str(MLSchool.target_function_vector.data[1]))




# Create a Python list that holds the names of the two columns.
# my_column_names = ['data', 'label']

# Create a DataFrame.
# my_dataframe = pd.DataFrame(data=data, columns=my_column_names)
# print(my_dataframe)
# print(my_dataframe['data'])