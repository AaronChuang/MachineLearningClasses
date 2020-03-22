import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import random


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def translateX(self, distance):
        if distance <= 0:
            raise ValueError('must be positive')
        return Point(self.x + distance, self.y)

    def translateY(self, distance):
        if distance <= 0:
            raise ValueError('must be positive')
        return Point(self.x, self.y + distance)

    def translateZ(self, distance):
        if distance <= 0:
            raise ValueError('must be positive')
        return Point(self.x, self.y)

    def translate(self, vector):
        return Point(self.x + vector.x, self.y + vector.y)

    # 相對原點的向量
    def vector(self):
        return Vector(self.x, self.y)


class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return "vector(" + str(self.x) + "," + str(self.y) + ")"

    def length(self):
        return np.sqrt(np.square(self.x) + np.square(self.y))

    def dotProduct(self, other):
        return self.x * other.x + self.y * other.y

    def scale(self, factor):
        return Vector(self.x * factor, self.y * factor)

    def normalization(self):
        return self.scale(1/self.length())

    # 假設有過原點的直線, 計算y座標
    def calculate_y_coordinate(self, x_coordinate):
        return self.x * x_coordinate / -self.y

    # 向量加法
    def addition(self, vector):
        return Vector(self.x + vector.x, self.y + vector.y)

    @staticmethod
    def vectorBetween(fromPoint, toPoint):
        return Vector(toPoint.x - fromPoint.x, toPoint.y - fromPoint.y)


class MLSchool:
    min = -100
    max = 100
    num_points = 100
    precision = 0.01
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
        for i in range(100):
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            self.points.append(Point(x, y))
        # for i in range(50):
        #     x = random.randint(-100, 0)
        #     y = random.randint(-100, 0)
        #     self.points.append(Point(x, y))

        for i in range(len(self.points)):
            group_no = MLSchool.get_group_no_by_group_condition(self.points[i])
            if group_no == 0:
                self.group1_points_x.append(MLSchool.points[i].x)
                self.group1_points_y.append(MLSchool.points[i].y)
                self.group1_points.append(MLSchool.points[i])
            else:
                self.group2_points_x.append(MLSchool.points[i].x)
                self.group2_points_y.append(MLSchool.points[i].y)
                self.group2_points.append(MLSchool.points[i])

    def draw(self, vector=None):
        if vector is not None:
            MLSchool.target_function_vector = vector.normalization()
        x = np.linspace(-100, 100, 400)
        y = (-MLSchool.target_function_vector.x / MLSchool.target_function_vector.y) * x -\
            MLSchool.function_const / MLSchool.target_function_vector.y
        plt.scatter(self.group1_points_x, self.group1_points_y, s=30, c='blue', label='blue')
        plt.scatter(self.group2_points_x, self.group2_points_y, s=30, c='red', label='red')
        plt.fill_between(x, y, 100, color='green', alpha=.5)
        plt.fill_between(x, y, -100, color='red', alpha=.5)
        plt.axis([-100, 100, -100, 100])
        plt.show()

    @staticmethod
    def get_group_no_by_group_condition(point):
        if point.x < 0 and point.y < 0:
            return 0
        else:
            return 1

    @staticmethod
    def get_function_number(point):
        return MLSchool.target_function_vector.x * point.x + MLSchool.target_function_vector.y * point.y + \
               MLSchool.function_const

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

    def perceptron_learning_algorithm(self):
        print('pla execute - '+str(MLSchool.calculate_times) + ',' +str(MLSchool.function_const))
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
                self.perceptron_learning_algorithm()
                break

    def check(self, group):
        cou = 0
        for i in range(len(group)):
            number = MLSchool.get_function_number(group[i])
            if number > 0:
                cou += 1
            print("("+str(group[i].x)+","+str(group[i].y)+") - " + "," +str(number))
        print(cou)


school = MLSchool()
school.init_data()
school.perceptron_learning_algorithm()
school.check(school.group1_points)
print("done - execute times:"+str(MLSchool.calculate_times))
print("done - function_const:" + str(MLSchool.function_const))
print("done - target_function_vector:" + str(MLSchool.target_function_vector.x) + "," + str(MLSchool.target_function_vector.y))

