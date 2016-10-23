#encoding: utf-8
import math
import numpy as np


class options:
    def __init__(self, discount_factor, temperature, epsilon, pmode):
        # 割引率
        self.discount_factor = discount_factor
        # softmaxの温度変数
        self.temprature = temperature
        # ε-greedyのepsilon
        self.epsilon = epsilon
        # 政策改善のmode 0:決定的な政策改善、1:ε-greedy、2: softmax
        self.pmode = pmode


def encoding_state(state3):
    '''
    encoding: 0:空, 1：◯、2：☓

    example)
    1,0,0,0,0,0,0,2 = ◯空空空空空空空☓ -> 1 * 3_0 + 0 * 3_1 + ... + 2 * 3_8

    :param state3: 9 dim numpy array
    :return:
    '''
    if len(state3) is not 9:
        assert("[Error] state3 vector is required 9 dim.")

    convert = np.array((
        (0,1,2,3,4,5,6,7,8),
        (2,1,0,5,4,3,8,7,6),
        (6,3,0,7,4,1,8,5,2),
        (0,3,6,1,4,7,2,5,8),
        (8,7,6,5,4,3,2,1,0),
        (6,7,8,3,4,5,0,1,2),
        (2,5,8,1,4,7,0,3,6),
        (8,5,2,7,4,1,6,3,0),
    ))

    power = np.array((
        math.pow(3,8),
        math.pow(3,7),
        math.pow(3,6),
        math.pow(3,5),
        math.pow(3,4),
        math.pow(3,3),
        math.pow(3,2),
        math.pow(3,1),
        math.pow(3,0),
    ))

    candidates = []
    for i in xrange(len(convert)):
        candidates.append(np.dot([state3[convert[i][j]] for j in range(len(convert[i]))], power))
    return min(candidates)


if __name__ == '__main__':
    state3_vec = np.array((0, 0, 0, 0, 0, 0, 0, 0, 0))
    print("0,0,0,0,0,0,0,0,0 endocing-> %d" % encoding_state(state3_vec))

    state3_vec = np.array((0, 0, 0, 0, 0, 0, 0, 0, 1))
    print("0,0,0,0,0,0,0,0,1 endocing-> %d" % encoding_state(state3_vec))
    state3_vec = np.array((1, 0, 0, 0, 0, 0, 0, 0, 0))
    print("1,0,0,0,0,0,0,0,0 endocing-> %d" % encoding_state(state3_vec))

    state3_vec = np.array((2, 0, 1, 2, 0, 0, 1, 0, 0))
    print("2,0,1,2,0,0,1,0,1 endocing-> %d" % encoding_state(state3_vec))
    state3_vec = np.array((1, 0, 2, 0, 0, 2, 0, 0, 1))
    print("1,0,2,0,0,2,0,0,1 endocing-> %d" % encoding_state(state3_vec))