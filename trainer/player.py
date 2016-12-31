# encoding: utf-8

class Player:
    def __init__(self, mark, policy=None):
        self.mark = mark
        self.policy = policy

    def update_env(self, env):
        self.env = env

    def select_index(self, current_state, step):
        print("Player: %s" % (self.mark.to_string()))
        actions = current_state.get_valid_actions()
        while(True):
            print("Please select index %s" % (actions))
            input_line = raw_input()

            if input_line == '':
                continue
            elif not input_line.isdigit():
                continue
            if int(input_line) in actions:
                return int(input_line)

    def learn(self, reward, state=None, action=None, next_state=None):
        pass