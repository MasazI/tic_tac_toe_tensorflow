#encoding: utf-8
import numpy as np
import random
import utils
from utils import game_check_state3

class TicTacToe():
    def __init__(self, n_states, n_actions, n_episodes, n_steps, options):
        '''
        ゲームの初期化
        :param n_states:
        :param n_actions:
        :param n_episodes:
        :param n_steps:
        :param options:
        '''
        self.n_states = n_states
        self.n_actions = n_actions
        self.n_episodes = n_episodes
        self.n_steps = n_steps

        # Q関数の初期化(ndarray)
        self.q = np.zeros((n_states, n_actions))
        # ポリシーの初期化
        self.policy = np.zeros([self.n_states, self.n_actions])

        self.options = options

    def initial_iteration(self):
        '''
        政策反復の初期化
        :return:
        '''

        self.visits = np.ones((self.n_states, self.n_actions), dtype=np.int32)
        self.results = np.zeros([self.n_episodes, 1], dtype=np.int32)
        self.states = np.zeros([self.n_episodes, self.n_steps], dtype=np.int32)
        self.states[self.states==0] = -1
        self.actions = np.empty([self.n_episodes, self.n_steps], dtype=np.int32)
        self.rewards = np.empty([self.n_episodes, self.n_steps])
        self.drewards = np.empty([self.n_episodes, self.n_steps])

    def initial_episode(self):
        '''
        エピソードの初期化（状態を初期化する）
        :return:
        '''
        # 状態の初期化
        self.state3 = np.zeros([self.n_actions])
        return self.state3

    def update_episodes_by_step(self, i_episode, i_step, state, action, reward):
        '''
        ステップごとのエピソード情報の更新
        :param i_episode:
        :param i_step:
        :param state:
        :param action:
        :param reward:
        :return:
        '''
        self.states[i_episode, i_step] = state
        self.actions[i_episode, i_step] = action
        self.rewards[i_episode, i_step] = reward
        self.visits[state, action] += 1

    def finish_episode(self, fin, i_episode, i_step):
        '''
        ゲーム終了時の処理

        :param fin: ゲーム終了フラグ
        :param i_episode: エピソードインデックス
        :param i_step: ステップインデックス
        :return:
        '''
        # ゲーム終了の情報
        self.results[i_episode] = fin
        # 各ステップの報酬を取得
        self.drewards[i_episode, i_step] = self.rewards[i_episode, i_step]

        # 割引率を考慮した報酬和（収益）の計算
        for pstep in reversed(xrange(i_step)):
            self.drewards[i_episode, pstep] = self.options.discount_factor * \
                                              self.drewards[i_episode, pstep + 1]

    def update_value_function(self):
        '''
        価値観数の更新

        エピソードを遡って、状態と行動の組に対する報酬を計算する
        :return:
        '''
        for m in xrange(self.n_episodes):
            for t in xrange(self.states.shape[1]):
                s = self.states[m, t]
                a = self.actions[m, t]

                if s == -1:
                    break

                self.q[s, a] += self.drewards[m, t]

        # 状態と行動の訪問数で割る
        self.q = self.q / self.visits

    def update_policy_improvement(self):
        '''
        政策の改善
        エピソードを遡って、q関数を使って最大の報酬を得たアクションに更新
        :return:
        '''
        for m in xrange(self.n_episodes):
            for t in xrange(self.states.shape[1]):
                s = self.states[m, t]
                if s == -1:
                    break
                self.policy_improvement(self.q, s, self.policy[s], self.options)

    def policy_improvement(self, q, state, policy, options):
        '''
        政策改善
        :param q: q function
        :param state: 状態
        :param policy: 政策（状態stateにおけるポリシー）
        :param options: 政策改善オプションクラス
        :return:
        '''
        # 政策の改善（s最初は初期化）
        if options.pmode == 0:
            # greedy
            max_action = np.argmax(q[state])
            policy[max_action] = 1.0

        elif options.pmode == 1:
            # ε-greedy
            max_action = np.argmax(q[state])

            # 最大のアクションのみ大きな確率を設定
            policy = np.ones([self.n_actions]) * options.epsilon / float(self.n_actions)
            policy[max_action] = 1 - options.epsilon + options.epsilon / float(self.n_actions)

            self.policy[state] = policy

        elif options.pmode == 2:
            # softmax
            self.policy[state] = np.exp(q[state]/options.temprature) / np.sum(np.exp(q[state]/options.temprature))


    def get_policy(self, state):
        '''
        状態におけるpolicyを返す
        :param state: 状態
        :return:
        '''
        return self.policy[state]

    def get_win_rate(self):
        '''
        勝率を取得する
        勝ちエピソード/エピソード数
        :return:
        '''
        t = self.results.copy()
        t[self.results != 2] = 0
        t[self.results == 2] = 1

        return t.sum() / float(self.n_episodes)

    def action_select(self, i_step, state3, policy, verbose=False):
        '''
        現在の政策と状態をもとにゲームを進める
        :param i_step: 現在のステップ数
        :param state3: 現在の状態
        :param policy: 現在の状態にたいする政策
        :return: action,reward,state2,fin
        '''

        action_list = np.array([i for i in xrange(self.n_actions)])

        # プレイヤー2 ########################################### （強化学習するPlayer）
        if i_step == 0:
            # 初期はActionリストからランダムにチョイスする
            a = random.choice(action_list)
        else:
            # 政策に従って行動を選択（random要素を追加）
            if verbose:
                utils.print_state3(state3)

            if verbose:
                print policy

            while(True):
                valid_action_list = np.where(state3 == 0)[0]

                # 選択不可能な行動に0を設定して選択の順位を調整する
                valid_policy = np.array([])
                for i, p in enumerate(policy):
                    if i in valid_action_list:
                        valid_policy = np.append(valid_policy, policy[i])
                    else:
                        valid_policy = np.append(valid_policy, 0.0)

                if verbose:
                    print valid_policy

                a = np.argmax(valid_policy)

                # TODO softmaxに合わせた選択


                # 1割はrandomにチョイスする
                if random.random() < 0.1:
                    if verbose:
                        print("random choice!!")
                    a = random.choice(valid_action_list)

                # マスがあいていれば選択完了
                if state3[a] == 0:
                    break
        # 自分の行動
        action = a
        state3[a] = 2

        if verbose:
            print("my action: %d" % (action))
            utils.print_state3(state3)

        if verbose:
            utils.print_state3(state3)

        # 勝負がついたか確認
        fin = game_check_state3(state3)
        if fin == 2:
            reward = 10.0
            return action, reward, state3, fin
        elif fin == 3:
            reward = 0.0
            return action, reward, state3, fin

        # プレイヤー1 （AI）###########################################
        reach = 0

        for position in utils.FIN_STATE:
            position_state = np.array([state3[i] for i in position])

            # プレイヤー1(AI)のマス
            player1_num= len(np.where(position_state == 1)[0])

            # 0 （空いているマス）を数える
            empty_num = len(np.where(position_state == 0)[0])

            if player1_num == 2 and empty_num == 1:
                # AIがリーチである
                reach = 1
                # 空いているマスのインデックスを使う
                a2 = position[np.where(position_state == 0)[0][0]]
                if verbose:
                    print("reach")
                    print position
                    print np.where(position_state == 0)
                    print a2
                break

        if reach == 0:
            if verbose:
                print("not reach")
            # リーチではない場合
            while(True):
                # ランダムにチョイス
                a2 = random.choice(action_list)

                # マスがあいていれば選択完了
                if state3[a2] == 0:
                    break
        state3[a2] = 1

        if verbose:
            utils.print_state3(state3)


        fin = game_check_state3(state3)
        if fin == 1:
            reward = -10.0
            return action, reward, state3, fin
        elif fin == 3:
            reward = 0.0
            return action, reward, state3, fin

        reward = 0.0
        return action, reward, state3, fin