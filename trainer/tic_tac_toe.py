# encoding: utf-8
'''
tic tac toe 手動ゲーム対戦
'''
import sys

from mark import Mark
from maru_mark import Maru
from batsu_mark import Batsu

from player import Player
from player_ql import QLearningPlayer
from tic_tac_toe_game import Game
from tic_tac_toe_state import State

from decision_ql import QLearningDecisionPolicy


def train():
    state = State()
    actions = state.get_valid_actions()
    print("valid actions: %s" % actions)
    state_array = state.to_array()
    print("state dim: %d" % len(state_array))

    input_dim = len(state_array) + 1
    policy_1 = QLearningDecisionPolicy(actions, input_dim, "train20161226-002008", "player1")
    policy_2 = QLearningDecisionPolicy(actions, input_dim, "train20161226-002008", "player2")

    com_1 = QLearningPlayer(Mark(Maru()), policy_1, True)
    com_2 = QLearningPlayer(Mark(Batsu()), policy_2, True)

    iterations = 1000

    for i in xrange(iterations):
        print("Game No.%d *******************************************" % i)
        game = Game(com_1, com_2)
        # print("[%d]" % (i))
        # print("="*30)
        game.start(i)
        if i % 1000 == 0:
            print("training iterations: No.%d" % (i))

    print("Test playing...")
    com_1.training = False
    com_2.training = False

    while (True):
        print("Select a type of fight [1, 2, 3, q]")
        print("1: human vs com2")
        print("2: com1 vs human")
        print("3: com1 vs com2")
        print("q: quit")

        type_of_fight = 1
        input_line = raw_input()
        if input_line.isdigit():
            type_of_fight = int(input_line)
        else:
            if input_line == 'q':
                break
            continue

        if type_of_fight == 1:
            game = Game(Player(Mark(Maru())), com_2)
        elif type_of_fight == 2:
            game = Game(com_1, Player(Mark(Batsu())))
        elif type_of_fight == 3:
            game = Game(com_1, com_2)
        game.start(verbose=True, step=0)


if __name__ == '__main__':
    argvs = sys.argv
    argc = len(argvs)
    if argc < 2:
        print("[Usage]%s <option (0: train com, 1: only human, 2: human vs com)>")
        sys.exit(1)
    try:
        option = int(argvs[1])
    except Exception as e:
        print("[Usage]%s <option (0: train com, 1: only human, 2: human vs com)>")
        sys.exit(1)

    if option == 0:
        print("Start train...")
        train()

    elif option == 1:
        print("Start Game...")
        player1 = Player(Mark(Maru()))
        player2 = Player(Mark(Batsu()))
        game = Game(player1, player2)
        game.start(verbose=True, step=0)
    elif option == 2:
        pass

    else:
        print("[Usage]%s <option (0: train com, 1: only human, 2: human vs com)>")
