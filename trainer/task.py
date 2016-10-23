#encoding: utf-8
import math
import numpy as np

import utils

N_STATES = math.pow(3, 9) #number of states
N_ACTIONS = 9 # number of actions
T = 5 # ax step number

def train_montecarlo(iteration_num, episode_num, options):
    '''
    :param interation_num: 政策反復回数
    :param eposode_num: エピソード
    :param options: optionsオブジェクト
    :return:
    '''
    q = np.zero((N_STATES, N_ACTIONS))

    for l in xrange(iteration_num):
        visits = np.ones((N_STATES, N_ACTIONS))
        results = np.zeros([episode_num, 1])

        for m in xrange(episode_num):
            # 状態の初期化
            state3 = np.zeros([N_ACTIONS])

            for t in xrange(T):
                # 状態のエンコード
                state = utils.encoding_state(state3)

                # 政策の初期化
                policy = np.zeros([N_ACTIONS])

                if options.pmode == 0:
                    # greedy
                    max_action = np.argmax(q[state])
                    policy[max_action] = 1.0

                elif options.pmode == 1:
                    # ε-greedy
                    max_action = np.argmax(q[state])
                    policy = np.ones([N_ACTIONS]) * options.epsilon / N_ACTIONS
                    policy[max_action] = 1 - options.epsilon + options.epsilon / N_ACTIONS

                elif options.pmode == 2:
                    # softmax









def main():
    train_montecarlo()


def __name__ == '__main__':
    main()