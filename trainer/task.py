#encoding: utf-8
import math
import numpy as np

import utils
from utils import Options
from tic_tac_toe import TicTacToe

N_STATES = int(math.pow(3, 9)) #number of states
N_ACTIONS = 9 # number of actions
T = 5 # max step number

def train_montecarlo(iteration_num, episode_num, options):
    '''
    :param interation_num: 政策反復回数
    :param eposode_num: エピソード
    :param options: optionsオブジェクト
    :return:
    '''
    tictactoe = TicTacToe(N_STATES, N_ACTIONS, episode_num, T, options)
    for l in xrange(iteration_num):
        print("iteration No.%d" % (l))
        tictactoe.initial_iteration()

        # TステップのエピソードをM回シュミレーション(標本集合)
        for m in xrange(episode_num):
            # 状態の初期化
            state3 = tictactoe.initial_episode()

            if m % 500 == 0:
                print("episode No.%d" % m)

            for t in xrange(T):
                # 状態のエンコード
                state = utils.encoding_state(state3)

                # 状態におけるポリシーを取得
                policy = tictactoe.get_policy(state)

                # 行動の選択と実行、実行後の状態ベクトルと報酬
                if m % 500 == 0:
                    verbose = True
                else:
                    verbose = False

                action, reward, state3, fin = tictactoe.action_select(t, state3, policy, verbose)

                # 状態、行動、報酬、出現回数の更新
                tictactoe.update_episodes_by_step(m, t, state, action, reward)

                # ゲームが終了したら割引率を考慮した報酬和（収益）を計算
                if fin > 0:
                    tictactoe.finish_episode(fin, m, t)
                    break

        # 状態行動価値関数の更新
        tictactoe.update_value_function()

        # 政策の改善
        tictactoe.update_policy_improvement()

        # 勝率計算
        print tictactoe.get_win_rate()


def main():
    options = Options(0.9, 0, 0.1, 1)
    train_montecarlo(30, 1000, options)


if __name__ == '__main__':
    main()