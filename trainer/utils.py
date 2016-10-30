#encoding: utf-8
import math
import numpy as np

FIN_STATE = (
    (0,1,2),
    (3,4,5),
    (6,7,8),
    (0,3,6),
    (1,4,7),
    (2,5,8),
    (0,4,8),
    (2,4,6),
)

BOARD = (
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),
)


class Options:
    def __init__(self, discount_factor, temperature, epsilon, pmode):
        # 割引率
        self.discount_factor = discount_factor
        # softmaxの温度変数
        self.temprature = temperature
        # ε-greedyのepsilon
        self.epsilon = epsilon
        # 政策改善のmode 0:決定的な政策改善、1:ε-greedy、2: softmax
        self.pmode = pmode


def game_check_state3(state3):
    '''
    ゲームが終了している状態かどうかの判定
    :param state3: 状態（ベクトル）
    :return: 0:続行 1:plyaer1の勝ち 2:player3の勝ち 3:引き分け
    '''
    for position in FIN_STATE:
        state_pos0 = state3[position[0]]
        state_pos1 = state3[position[1]]
        state_pos2 = state3[position[2]]

        # player 1 の勝ち
        if state_pos0 == 1 and state_pos1 == 1 and state_pos2 == 1:
            fin = 1
            return fin

        # player 2 の勝ち
        elif state_pos0 == 2 and state_pos1 == 2 and state_pos2 == 2:
            fin = 2
            return fin

    # 勝負がついていない
    for i in state3:
        if i == 0:
            fin = 0
            return fin

    # 空いたマスが無いかつプレイヤーの勝敗がつかない
    fin = 3
    return fin


def print_state3(state3):
    '''
    state3の盤面出力
    0 -> .
    1 -> o
    2 -> x
    :param state3:
    :return:
    '''
    for line in BOARD:
        print "-" * 11
        line_str = ""
        for i in line:
            if state3[i] == 0:
                line_str += " . "
            elif state3[i] == 1:
                line_str += " o "
            elif state3[i] == 2:
                line_str+= " x "
            line_str+= "|"
        print line_str
    print "-" * 11
    print "=" * 20

def encoding_state(state3):
    '''
    encoding: 0:空, 1：◯ (AI)、2：☓ (Player)

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
    state = int(min(candidates))
    return state


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