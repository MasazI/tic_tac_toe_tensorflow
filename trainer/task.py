#encoding: utf-8
import math
import numpy as np

import utils
from train import action_select

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

        states = np.empty([episode_num, T])
        actions = np.empty([episode_num, T])
        rewards = np.empty([episode_num, T])
        drewards = np.empty([episode_num, T])

        # TステップのエピソードをM回シュミレーション
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
                    # TODO implementation softmax
                    pass

                action, reward, state3, fin = action_select(policy, t, state3)

                # 状態、行動、報酬、出現回数の更新
                states[m, t] = state
                actions[m, t] = action
                rewards[m, t] = reward
                visits[state, action] += 1

                # ゲームが終了したら割引率を考慮した報酬和（収益）を計算
                if fin > 0:
                    results[m] = fin
                    drewards[m, t] = rewards[m, t]
                    for pstep in reversed(xrange(t)):
                        drewards[m, pstep] = options.discount_factor * drewards[m, pstep + 1]
                    break

        # 状態行動価値関数の更新
        Q = np.zeros([N_STATES, N_ACTIONS])
        for m in xrange(episode_num):
            for t in xrange(states.shape[1]):
                s = states[m, t]
                a = actions[m, t]
                if s == 0:
                    break
                Q[s, a] += drewards[m, t]
        Q = Q / visits

        # TODO 勝率計算





def main():
    train_montecarlo()


def __name__ == '__main__':
    main()