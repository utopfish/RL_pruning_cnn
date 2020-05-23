import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
class PolicyGradient:
    def __init__(self, n_actions, n_features, learning_rate=0.01, reward_decay=0.95):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate  # learning_rate
        self.gamma = reward_decay  # reward decay

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []  # info list

        self._build_net()  # build policy net

        self.sess = tf.Session()

        self.sess.run(tf.global_variables_initializer())

    def _build_net(self):
        with tf.name_scope('inputs'):
            self.tf_obs = tf.placeholder(tf.float32, [None, self.n_features], name="observations")  # get observation
            self.tf_acts = tf.placeholder(tf.int32, [None,], name="actions_num")  # get actions
            self.tf_vt = tf.placeholder(tf.float32, [None, ],
                                        name="actions_value")  # get state-action of value

        # fc1
        layer = tf.layers.dense(
            inputs=self.tf_obs,
            units=10,  # output number
            activation=tf.nn.tanh,  # activation function
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc1'
        )
        # fc2
        all_act = tf.layers.dense(
            inputs=layer,
            units=self.n_actions,  # output number
            activation=None,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc2'
        )

        self.all_act_prob = tf.nn.softmax(all_act, name='act_prob')  # softmax

        with tf.name_scope('loss'):
            #交叉熵
            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=all_act, labels=self.tf_acts)
            #与归一化后的reward相乘，加-求最大值
            loss = -tf.reduce_mean(neg_log_prob * self.tf_vt)  #
        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)

    def choose_action(self, observation):

        prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.tf_obs: observation[np.newaxis, :]})    # 所有 action 的概率
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())  # 根据概率来选 action
        return action

    def store_transition(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def learn(self):
        #  normalize reward
        discounted_ep_rs_norm = self._discount_and_norm_rewards()

        # train on episode
        self.sess.run(self.train_op, feed_dict={
            self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
            self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
            self.tf_vt: discounted_ep_rs_norm,  # shape=[None, ]
        })
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []  # 清空回合 data
        return discounted_ep_rs_norm  # return state-action value

    def _discount_and_norm_rewards(self):
        # discount episode rewards
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add

        # normalize episode rewards
        if len(discounted_ep_rs)==1:
            return discounted_ep_rs
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs
