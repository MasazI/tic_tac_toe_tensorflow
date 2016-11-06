#encoding: utf-8
import math
import numpy as np

import utils
from utils import Options
from tic_tac_toe import TicTacToe

N_STATES = int(math.pow(3, 9)) #number of states
N_ACTIONS = 9 # number of actions
T = 10 # max step number

def train_montecarlo(iteration_num, episode_num, options, debug=False):
    '''
    :param interation_num: 政策反復回数
    :param eposode_num: エピソード
    :param options: optionsオブジェクト
    :param debug: True:debug, False:Run
    :return:
    '''
    winrates = []

    # ゲームと学習するQを初期化
    tictactoe = TicTacToe(N_STATES, N_ACTIONS, episode_num, T, options)

    for l in xrange(iteration_num):
        if l % 1000 == 0:
            print(">>"*100)
            print("iteration No.%d" % (l))
        tictactoe.initial_iteration()

        # TステップのエピソードをM回シュミレーション(標本集合)
        for m in xrange(episode_num):
            # 状態の初期化
            state3 = tictactoe.initial_episode()

            if l % 1000 == 0:
                print("*"*100)
                print("episode No.%d" % m)

            for t in xrange(T):
                # 状態のエンコード
                state = utils.encoding_state(state3)

                # 状態におけるポリシーを取得
                policy = tictactoe.get_policy(state)

                # 行動の選択と実行、実行後の状態ベクトルと報酬
                if l % 1000 == 0:
                    verbose = debug
                else:
                    verbose = False

                action, reward, state3, fin = tictactoe.action_select(t, state3, policy, verbose)

                if l % 1000 == 0:
                    print("reward %d" % (reward))

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
        if l % 1000 == 0:
            print("Iteration No.%d win rate: %f" % (l, tictactoe.get_win_rate()))
        winrates.append(tictactoe.get_win_rate())
    return winrates



def main():
    options = Options(0.9, 0.1, 0.1, 2)
    winrates = train_montecarlo(iteration_num=10000, episode_num=1000, options=options, debug=True)

    print winrates

if __name__ == '__main__':
    main()