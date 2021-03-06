#encoding: utf-8
from player import Player
import numpy as np
import random

class QLearningPlayer(Player):
    epsilon = 0.1

    def __init__(self, mark, policy, training=True):
        self.mark = mark
        self.policy = policy

        self.training = training
        # 1つ前の状態の報酬（ゲーム終了次に使う）
        self.previous_reward = None
        # 1つ前の事項状態（状態と意味は同じだが、ゲームのように相手によって状態が変わる場合には区別する）
        # 1つ前の事故状態に関する価値の更新を、今回の行動の価値によって行う（行動選択の後に行う）
        self.previous_after_state = None

    def update_env(self, env):
        self.env = env

    def select_index(self, state, step):
        mark_int = self.mark.to_int()
        state_input = np.asmatrix(np.hstack((mark_int, state.to_array())))

        # select action
        valid_actions = state.get_valid_actions()
        action = self.policy.select_action(state_input, step, valid_actions=valid_actions)
        selected_action = action

        if self.training:
            after_state = state.set(action, self.mark)

            if (self.previous_reward is not None) and (self.previous_after_state is not None):
                mark_int = self.mark.to_int()
                previous_after_state_input = np.asmatrix(np.hstack((mark_int, self.previous_after_state.to_array())))
                # create input of neural network
                after_state_input = np.asmatrix(np.hstack((mark_int, after_state.to_array())))

                print("update using previous_reward: %d %d" % (self.previous_reward, self.mark.to_int()))
                print("mark: %s" % (self.mark.to_string()))
                print("privious state =====")
                print self.previous_after_state.output()
                print("current state ======")
                print state.output()
                print("after state ========")
                print after_state.output()
                print("-"*20)

                self.policy.update_q(previous_after_state_input, self.previous_reward, after_state_input, valid_actions=valid_actions)
            else:
                pass

            # soft-policy
            if random.random() < self.epsilon:
                selected_action = random.choice(state.get_valid_actions())

            self.previous_after_state = state.set(selected_action, self.mark)

        return action

    def learn(self, reward, state=None, next_state=None, finish=False):
        if self.training:
            if finish:
                if state is not None:
                    print("change previous after state")
                    # 負け、もしくは引き分けの場合、勝敗が決定した時点でのstateに更新してから敗者のQ(s,a)を更新する
                    #self.previous_after_state = state

                mark_int = self.mark.to_int()
                previous_after_state_input = np.asmatrix(np.hstack((mark_int, self.previous_after_state.to_array())))

                if next_state is not None:
                    next_state_input = np.asmatrix(np.hstack((mark_int, next_state.to_array())))
                else:
                    next_state_input = next_state

                print("Finish!!! update using reward: %d %d" % (reward, self.mark.to_int()))
                print("privious after state =============")
                print self.previous_after_state.output()
                print("next state input.")
                print next_state_input
                print("-" * 20)

                self.policy.update_q(previous_after_state_input, reward, next_state_input)
                self.previous_reward = None
                self.previous_after_state = None
            else:
                self.previous_reward = reward