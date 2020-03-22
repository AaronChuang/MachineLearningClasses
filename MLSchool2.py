import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import random


class Point:

    def __init__(self, data):
        self.data = data

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
        self.data = data

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
            new_data[i] = new_data.data[i] - fromPoint.data[i]
        return Vector(new_data)


class MLSchool:
    min = -100
    max = 100
    num_points = 100
    precision = 0.01
    dimension = 2
    points = []
    group1_points = []
    group2_points = []
    group1_points_x = []
    group2_points_x = []
    group1_points_y = []
    group2_points_y = []
    target_function_vector = None
    calculate_times = 0
    sign = lambda a: MLSchool.group_sign[0] if a < 0 else MLSchool.group_sign[1]
    group_sign = [-1, 1]
    function_const = 0

    def init_data(self):
        random.seed('magic')
        # 初始化數據
        for i in range(50):
            data = []
            for i in range(self.dimension):
                data.append(random.randint(1, 100))
            self.points.append(Point(data))
        for i in range(50):
            data = []
            for i in range(self.dimension):
                data.append(random.randint(-100, -1))
            self.points.append(Point(data))
        for i in range(len(self.points)):
            group_no = MLSchool.get_group_no_by_group_condition(self.points[i])
            if group_no == 0:
                self.group1_points_x.append(MLSchool.points[i].data[0])
                self.group1_points_y.append(MLSchool.points[i].data[1])
                self.group1_points.append(MLSchool.points[i])
            else:
                self.group2_points_x.append(MLSchool.points[i].data[0])
                self.group2_points_y.append(MLSchool.points[i].data[1])
                self.group2_points.append(MLSchool.points[i])

    def draw(self, vector=None):
        if vector is not None:
            MLSchool.target_function_vector = vector.normalization()
        x = np.linspace(-100, 100, 400)
        y = (-MLSchool.target_function_vector.data[0] / MLSchool.target_function_vector.data[1]) * x -\
            MLSchool.function_const / MLSchool.target_function_vector.data[1]
        plt.scatter(self.group1_points_x, self.group1_points_y, s=30, c='blue', label='blue')
        plt.scatter(self.group2_points_x, self.group2_points_y, s=30, c='red', label='red')
        plt.fill_between(x, y, 100, color='green', alpha=.5)
        plt.fill_between(x, y, -100, color='red', alpha=.5)
        plt.axis([-100, 100, -100, 100])
        plt.show()

    @staticmethod
    def get_group_no_by_group_condition(point):
        check = True
        for i in range(len(point.data)):
            if point.data[i] > 0:
                check = False
        if check:
            return 0
        else:
            return 1

    @staticmethod
    def get_function_number(point):
        result = MLSchool.function_const
        for i in range(MLSchool.dimension):
            result = result + MLSchool.target_function_vector.data[i] * point.data[i]
        return result

    def check_sign_fail(self, point, predict_sign):
        number = MLSchool.get_function_number(point)
        if MLSchool.sign(number) == predict_sign:
            return False
        for i in np.arange(self.min, self.max, self.precision):
            MLSchool.function_const = i
            number = MLSchool.get_function_number(point)
            if MLSchool.sign(number) == predict_sign:
                return False
        return True

    def PLA(self):
        print('pla execute - '+str(MLSchool.calculate_times) + "," + str(MLSchool.target_function_vector) + ',' + str(MLSchool.function_const))
        if MLSchool.target_function_vector is None:
            MLSchool.target_function_vector = self.points[0].vector().normalization()
        MLSchool.calculate_times += 1
        self.draw()
        for i in range(len(self.points)):
            group_no = MLSchool.get_group_no_by_group_condition(self.points[i])
            if self.check_sign_fail(self.points[i], self.group_sign[group_no]):
                MLSchool.target_function_vector = MLSchool.target_function_vector\
                    .addition(self.points[i].vector())\
                    .normalization()
                self.PLA()
                break

    def check(self, group):
        cou = 0
        for i in range(len(group)):
            number = MLSchool.get_function_number(group[i])
            if number > 0:
                cou += 1
            print("("+str(group[i].data[0])+","+str(group[i].data[1])+") - " + "," +str(number))
        print(cou)


school = MLSchool()
school.init_data()
school.PLA()
school.check(school.group1_points)
print("done - execute times:"+str(MLSchool.calculate_times))
print("done - function_const:" + str(MLSchool.function_const))
print("done - target_function_vector:" + str(MLSchool.target_function_vector.data[0]) + "," + str(MLSchool.target_function_vector.data[1]))

