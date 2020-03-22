import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import random


class Point:

    def __init__(self, data):
        self.data = data

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
            new_data[i] = new_data[i] - fromPoint.data[i]
        return Vector(new_data)


class MLSchool:
    min = -100
    max = 100
    num_points = 100
    precision = 0.01
    dimension = 2
    # pocket algo
    choosePoint1 = -1
    choosePoint2 = -1
    target_function_points_num = -1
    # pocket algo end
    points = []
    group1_points = []
    group2_points = []
    group1_points_x = []
    group2_points_x = []
    group1_points_y = []
    group2_points_y = []
    target_function_vector = None
    target_function_const = 0
    tmp_function_const = 0
    calculate_times = 0
    sign = lambda a: MLSchool.group_sign[0] if a < 0 else MLSchool.group_sign[1]
    group_sign = [-1, 1]

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

    def draw(self):
        x = np.linspace(-100, 100, 400)
        y = MLSchool.target_function_const
        if MLSchool.target_function_vector.data[1] != 0:
            y = (-MLSchool.target_function_vector.data[0] / MLSchool.target_function_vector.data[1]) * x -\
                MLSchool.target_function_const / MLSchool.target_function_vector.data[1]

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
    def get_function_number(point, vector=None, const=None):
        if vector is None:
            vector = MLSchool.target_function_vector
        if const is None:
            const = MLSchool.target_function_const
        result = const
        for i in range(MLSchool.dimension):
            result = result + vector.data[i] * point.data[i]
        return result

    def check_sign_fail(self, point, predict_sign):
        number = MLSchool.get_function_number(point)
        if MLSchool.sign(number) != predict_sign:
            # 嘗試更換常數項, 更換後更新常數並重繪製
            for i in np.arange(self.min, self.max, self.precision):
                self.tmp_function_const = i
                number = MLSchool.get_function_number(point, None,  self.tmp_function_const)
                if MLSchool.sign(number) == predict_sign:
                    MLSchool.target_function_const = self.tmp_function_const
                    break
            # 更新向量
            MLSchool.target_function_vector = MLSchool.target_function_vector \
                .addition(point.vector()) \
                .normalization()
            return True
        return False

    def perceptron_learning_algorithm(self):
        print('pla execute - '+str(MLSchool.calculate_times) + "," + str(MLSchool.target_function_vector) + ',' + str(MLSchool.target_function_const))
        if MLSchool.target_function_vector is None:
            MLSchool.target_function_vector = self.points[0].vector().normalization()
        MLSchool.calculate_times += 1
        self.draw()
        for i in range(len(self.points)):
            group_no = MLSchool.get_group_no_by_group_condition(self.points[i])
            if self.check_sign_fail(self.points[i], self.group_sign[group_no]):
                self.perceptron_learning_algorithm()
                break

    def has_next_main_point(self):
        return self.choosePoint1 < (len(self.points) - 1)

    def next_main_point(self):
        self.choosePoint1 = self.choosePoint1 + 1

    def has_next_end_point(self):
        return self.choosePoint2 < len(self.points)

    def next_end_point(self):
        self.choosePoint2 = self.choosePoint2 + 1

    def reset_end_point(self):
        self.choosePoint2 = self.choosePoint1

    def get_vector(self):
        p1 = self.points[self.choosePoint1]
        p2 = self.points[self.choosePoint2]
        return Vector.vectorBetween(p1, p2)

    def check_sign_fail_packet_algo(self, point, predict_sign):
        number = MLSchool.get_function_number(point)
        if MLSchool.sign(number) != predict_sign:
            # 嘗試更換常數項, 更換後更新常數並重繪製
            for i in np.arange(self.min, self.max, self.precision):
                self.tmp_function_const = i
                number = MLSchool.get_function_number(point, None,  self.tmp_function_const)
                if MLSchool.sign(number) == predict_sign:
                    MLSchool.target_function_const = self.tmp_function_const
                    break
            # 更新向量
            MLSchool.target_function_vector = MLSchool.target_function_vector \
                .addition(point.vector()) \
                .normalization()
            return True
        return False

    def check_all_points(self, vector):
        result = -1
        # 常數變化

        for i in np.arange(self.min, self.max, self.precision):
            local_function_const = i
            temp_wrong = 0
            # 找最少錯誤點
            for j in range(len(self.points)):
                group_no = MLSchool.get_group_no_by_group_condition(self.points[j])
                number = MLSchool.get_function_number(self.points[j], vector, local_function_const)
                # print("("+str(self.points[j].data[0])+","+str(self.points[j].data[1])+") - " + "," +str(number) +
                #       ","+str(group_no)+","+str(self.group_sign[group_no])+","+str(MLSchool.sign(number)))
                if MLSchool.sign(number) != self.group_sign[group_no]:
                    temp_wrong = temp_wrong + 1
            if result == -1 or result > temp_wrong:
                result = temp_wrong
                self.tmp_function_const = local_function_const
        return result

    def pocket_algorithm(self):
        # 有下個起點且錯誤數量不為零則繼續找
        while self.has_next_main_point() and MLSchool.target_function_points_num != 0:
            print('pocket_algorithm execute - '+str(MLSchool.calculate_times) + "," +
                  str(MLSchool.target_function_vector) + ',' + str(MLSchool.target_function_const))
            self.next_main_point()
            self.reset_end_point()
            # 有下個終點且錯誤數量不為零則繼續找
            while self.has_next_end_point() and MLSchool.target_function_points_num != 0:
                MLSchool.calculate_times += 1
                self.next_end_point()
                vector = self.get_vector().normalization()
                if MLSchool.target_function_vector is None:
                    MLSchool.target_function_vector = vector
                wrongs = self.check_all_points(vector)
                print("wrongs ="+str(wrongs))
                if wrongs < MLSchool.target_function_points_num or MLSchool.target_function_points_num == -1:
                    MLSchool.target_function_vector = vector
                    MLSchool.target_function_points_num = wrongs
                    MLSchool.target_function_const = self.tmp_function_const
                    self.draw()

    def check(self, group):
        cou = 0
        for i in range(len(group)):
            number = MLSchool.get_function_number(group[i])
            if number > 0:
                cou += 1
            no = MLSchool.get_group_no_by_group_condition(group[i])
            print("check ("+str(group[i].data[0])+","+str(group[i].data[1])+") - " +str(number)+","+str(self.group_sign[no]))
        print(cou)


school = MLSchool()
school.init_data()
# school.perceptron_learning_algorithm()
school.pocket_algorithm()
school.check(school.group1_points)
print("XXXXXXX")
school.check(school.group2_points)

print("done - execute times:"+str(MLSchool.calculate_times))
print("done - function_const:" + str(MLSchool.target_function_const)+","+str(MLSchool.target_function_points_num))
print("done - target_function_vector:" + str(MLSchool.target_function_vector.data[0]) + "," + str(MLSchool.target_function_vector.data[1]))

