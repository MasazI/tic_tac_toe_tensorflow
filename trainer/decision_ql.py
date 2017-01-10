#encoding: utf-8
import tensorflow as tf
import numpy as np
from decision_inf import DecisionPolicy
from model import mlp
import random
from tensorflow.python.platform import gfile

iteration_num = 10000

class QLearningDecisionPolicy(DecisionPolicy):
    def __init__(self, actions, input_dim, model_dir, scope_name="mlp0"):
        '''
        :param actions: 行動（NNの出力）
        :param input_dim: NNの入力次元
        :param model_dir: モデル保存ディレクトリ
        '''
        # q-learning rate
        self.qlr = 0.1
        # select action function hyper-parameters
        self.epsilon = 0.9
        # q functins hyper-parameters
        self.gamma = 0.99
        # neural network hyper-parmetrs
        self.lr = 0.01

        self.actions = actions
        output_dim = len(actions)

        # neural network input and output placeholder
        self.x = tf.placeholder(tf.float32, [None, input_dim])
        self.y = tf.placeholder(tf.float32, [output_dim])

        # hidden layer dimension
        h1_dim = 10

        # model inference
        self.q = mlp(scope_name, self.x, input_dim, h1_dim, output_dim)

        # loss
        loss = tf.square(self.y - self.q)

        # train operation
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)

        # session
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        # initalize
        init_op = tf.initialize_all_variables()
        self.sess.run(init_op)

        # saver
        self.saver = tf.train.Saver(tf.trainable_variables())

        # laod model
        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print("laod model: %s" % (ckpt.model_checkpoint_path))
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)

    def select_action(self, current_state, step, valid_actions=None):
        threshold = min(self.epsilon, step/float(iteration_num))

        print valid_actions
        print("threshold: %f" % (threshold))
        rand = random.random()
        print("rand: %f" % (rand))

        if rand > threshold:
            # random choice
            print("random choice.")
            action = valid_actions[random.randint(0, len(valid_actions) - 1)]
        else:
            print("decision from Q function.")
            action_q_vals = self.sess.run(self.q, feed_dict={self.x: current_state})

            print action_q_vals
            #condition of valid_actions
            ind = np.ones(action_q_vals[0].shape, dtype=bool)
            ind[valid_actions] = False
            action_q_vals[0][ind] = 0.0
            action_idx = np.argmax(action_q_vals[0])
            if action_idx in valid_actions:
                action = self.actions[action_idx]
            else:

                action = valid_actions[random.randint(0, len(valid_actions) - 1)]

        return action

    def update_q(self, state, reward, next_state, valid_actions=None):
        if next_state is not None:
            # Q(s, a)
            action_q_vals = self.sess.run(self.q, feed_dict={self.x: state})
            # Q(s', a')
            next_action_q_vals = self.sess.run(self.q, feed_dict={self.x: next_state})

            # a' index with condition of valid actions
            # ind = np.ones(next_action_q_vals[0].shape, dtype=bool)
            # ind[valid_actions] = False
            # next_action_q_vals[0][ind] = 0.0
            # next_action_idx = np.argmax(next_action_q_vals[0])

            # a' index
            # next_action_idx = np.argmax(next_action_q_vals[0])

            # create target
            # action_q_vals[0, next_action_idx] += self.qlr*(reward +
            #                                               self.gamma * next_action_q_vals[0, next_action_idx] -
            #                                               action_q_vals[0, next_action_idx])
            action_q_vals[0] += self.qlr*(reward + self.gamma * next_action_q_vals[0] - action_q_vals[0])
            # delete minibatch dim
            action_q_vals = np.squeeze(np.asarray(action_q_vals))

            # train
            self.sess.run(self.train_op, feed_dict={self.x: state, self.y: action_q_vals})
        else:
            # Q(s, a)
            action_q_vals = self.sess.run(self.q, feed_dict={self.x: state})
            action_q_vals[0] += self.qlr * (reward - action_q_vals[0])
            # delete minibatch dim
            action_q_vals = np.squeeze(np.asarray(action_q_vals))
            # train
            self.sess.run(self.train_op, feed_dict={self.x: state, self.y: action_q_vals})


    def save_model(self, output_dir, step):
        if not gfile.Exists(output_dir):
            gfile.MakeDirs(output_dir)

        checkpoint_path = output_dir + '/model.ckpt'
        self.saver.save(self.sess, checkpoint_path, global_step=step)

