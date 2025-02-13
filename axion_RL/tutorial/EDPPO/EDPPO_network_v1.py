# EDPPO Network
# environment setting
import os
import datetime
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Lambda, Conv2D, BatchNormalization, Activation, Flatten, MaxPooling2D
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import ray

# from sklearn.preprocessing import minmax_scale
# import tensorflow_probability as tfp
# tfd = tfp.distributions

@ray.remote(num_gpus=1)
class Network(object):
    """
        Network for EDPPO
    """
    def __init__(self, state_shape, action_shape, design_length_x, design_length_y, actor_learning_rate, critic_learning_rate, ratio_clipping, batch_size):
        self.state_shape = state_shape  # ndarray:(unit_num_x, unit_num_y, 3)
        self.action_shape = action_shape  # ndarray:(unit_num_x, unit_num_y, 1)
        self.network_out_shape = np.zeros((25, 1))
        self.design_length_x = design_length_x  # 생성 블럭의 한 변 길이 (8.0)
        self.design_length_y = design_length_y  # 생성 블럭의 한 변 길이 (4.0)
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.ratio_clipping = ratio_clipping
        self.batch_size = batch_size

        self.std_bound = [1, 1e+1]  # std bound

        # # coverage_area / 확률함수 별 평균값 지정을 위한 함수 생성
        # self.indices_x, self.indices_y, self.update_add_x, self.update_add_y, self.coverage_length_x, self.coverage_length_y = self.coverage_area()

        # 계산 효율을 위한 디자인 노드 생성
        self.design_node, self.design_batch_node = self.design_area()

        ## create actor network
        self.actor = self.actor_build_network()
        self.actor_optimizer = Adam(self.actor_learning_rate)

        ## create critic network
        self.critic, self.states = self.critic_build_network()
        self.critic.compile(optimizer=Adam(self.critic_learning_rate), loss='mse')

        # 가중치 불러오기
        self.actor_load_weights('./save_weights/')
        self.critic_load_weights('./save_weights/')

        # 텐서보드 적용
        log_dir = self.make_Tensorboard_dir("Learning_log")
        self.writer = tf.summary.create_file_writer(log_dir)

        # GPU Check
        print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


    # 텐서보드 working directory 지정
    def make_Tensorboard_dir(self, dir_name):
        root_logdir = os.path.join(os.curdir, dir_name)
        sub_dir_name = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        return os.path.join(root_logdir, sub_dir_name)

    # 텐서보드 적용할 데이터 입력
    def draw_Tensorboard(self, score, step, episode, batch_action, batch_log_old_policy_pdf, gaes_sample, y_i_sample, mu_x, mu_y, sig_x, sig_y, wt):
        batch_action = tf.convert_to_tensor(batch_action)
        with self.writer.as_default():
            tf.summary.scalar('Duration / Episode', step, step=episode)
            tf.summary.scalar('Total Reward / Episode', score, step=episode)
            # tf.summary.scalar('is Reward? (n-step_target) / Episode', y_i_sample[0], step=episode)
            tf.summary.scalar('Reward - v_value (GAE) / Episode', gaes_sample[0], step=episode)
            tf.summary.scalar('current_log_policy_pdf / Episode', batch_log_old_policy_pdf[0][0], step=episode)
            tf.summary.histogram('Action Prob / Episode', batch_action, step=episode)

            tf.summary.histogram('mu_x / Episode', mu_x[0], step=episode)
            tf.summary.histogram('mu_y / Episode', mu_y[0], step=episode)
            tf.summary.histogram('sig_x / Episode', sig_x[0], step=episode)
            tf.summary.histogram('sig_y / Episode', sig_y[0], step=episode)
            tf.summary.histogram('wt / Episode', wt[0], step=episode)

    def debug_tensorboard(self, log_old_policy_pdf, plane_log_prob, ratio, loss, EPOCH):
        with self.writer.as_default():
            tf.summary.histogram('update_log_prob', plane_log_prob, step=EPOCH)
            tf.summary.histogram('old_log_prob', log_old_policy_pdf, step=EPOCH)
            tf.summary.histogram('ratio', ratio, step=EPOCH)
            tf.summary.scalar('loss', loss, step=EPOCH)


    ## actor network
    def actor_build_network(self):
        # test network _ input ( 80 x 40 x 1) -> output ( 25 x 6 )
        state_input = Input(shape=self.state_shape)
        h1 = Conv2D(64, 3, 1, 'same', kernel_initializer='he_normal', use_bias=False)(state_input)
        h1 = MaxPooling2D(2, strides=2, padding='valid')(h1)
        h1 = BatchNormalization()(h1)
        h1 = Activation('swish')(h1)
        h2 = Conv2D(192, 3, 1, 'same', kernel_initializer='he_normal', use_bias=False)(h1)
        # h2 = MaxPooling2D(2, strides=2, padding='valid')(h2)
        h2 = BatchNormalization()(h2)
        h2 = Activation('swish')(h2)
        h3 = Conv2D(192, 3, 1, 'same', kernel_initializer='he_normal', use_bias=False)(h2)
        h3 = BatchNormalization()(h3)
        h3 = Activation('swish')(h3)
        h4 = Conv2D(128, 3, 1, 'same', kernel_initializer='he_normal', use_bias=False)(h3)
        h4 = BatchNormalization()(h4)
        h4 = Activation('swish')(h4)
        h5 = Conv2D(128, 3, 1, 'same', kernel_initializer='he_normal', use_bias=False)(h4)
        h5 = MaxPooling2D(2, strides=2, padding='valid')(h5)
        h5 = BatchNormalization()(h5)
        h5 = Activation('swish')(h5)
        # h6 = Conv2D(128, 3, 1, 'same', kernel_initializer='he_normal')(h5)
        # h6 = BatchNormalization()(h6)
        # h6 = Activation('swish')(h6)
        # h7 = Conv2D(64, 3, 1, 'same', kernel_initializer='he_normal')(h6)
        # h7 = BatchNormalization()(h7)
        # h7 = Activation('swish')(h7)
        # h8 = Conv2D(128, 3, 1, 'same', kernel_initializer='he_normal')(h7)
        # h8 = BatchNormalization()(h8)
        # h8 = Activation('swish')(h8)
        # h9 = Conv2D(64, 3, 1, 'same', kernel_initializer='he_normal')(h8)
        # h9 = BatchNormalization()(h9)
        # h9 = Activation('swish')(h9)
        # h10 = Conv2D(32, 3, 1, 'same', kernel_initializer='he_normal')(h9)
        # h10 = MaxPooling2D(2, strides=2, padding='valid')(h10)
        # h10 = BatchNormalization()(h10)
        # h10 = Activation('swish')(h10)
        flat_layer = Flatten()(h5)

        weight_output = Dense(self.network_out_shape.size, activation='softmax', kernel_initializer='he_normal')(flat_layer)

        # out_mu_x = Dense(self.network_out_shape.size, activation='linear', kernel_initializer='he_normal')(flat_layer)
        # out_mu_x = Lambda(lambda x: tf.clip_by_value(x, clip_value_min=0., clip_value_max=self.design_length_x))(out_mu_x)
        #
        # out_mu_y = Dense(self.network_out_shape.size, activation='linear', kernel_initializer='he_normal')(flat_layer)
        # out_mu_y = Lambda(lambda x: tf.clip_by_value(x, clip_value_min=0., clip_value_max=self.design_length_y))(out_mu_y)

        out_mu_x = Dense(self.network_out_shape.size, activation='tanh', kernel_initializer='he_normal')(flat_layer)
        out_mu_x = Lambda(lambda x: self.design_length_x / 2. + tf.multiply(x, self.design_length_x / 2.))(out_mu_x)

        out_mu_y = Dense(self.network_out_shape.size, activation='tanh', kernel_initializer='he_normal')(flat_layer)
        out_mu_y = Lambda(lambda x: self.design_length_y / 2. + tf.multiply(x, self.design_length_y / 2.))(out_mu_y)

        std_output_x = Dense(self.network_out_shape.size, activation='sigmoid', kernel_initializer='he_normal')(flat_layer)
        std_output_x = Lambda(lambda x: x * (self.std_bound[1] - self.std_bound[0]) + self.std_bound[0])(std_output_x)

        std_output_y = Dense(self.network_out_shape.size, activation='sigmoid', kernel_initializer='he_normal')(flat_layer)
        std_output_y = Lambda(lambda x: x * (self.std_bound[1] - self.std_bound[0]) + self.std_bound[0])(std_output_y)

        corr_output_xy = Dense(self.network_out_shape.size, activation='tanh', kernel_initializer='he_normal')(flat_layer)
        # corr_output_xy = Lambda(lambda x: tf.clip_by_value(x, clip_value_min=-1., clip_value_max=1.))(corr_output_xy)

        model = Model(state_input, [out_mu_x, out_mu_y, std_output_x, std_output_y, corr_output_xy, weight_output])
        model.summary()

        return model

    ## critic network
    def critic_build_network(self):
        state_input = Input(shape=self.state_shape)
        h1 = Conv2D(64, 3, 1, 'same', kernel_initializer='he_normal', use_bias=False)(state_input)
        h1 = MaxPooling2D(2, strides=2, padding='valid')(h1)
        h1 = BatchNormalization()(h1)
        h1 = Activation('swish')(h1)
        h2 = Conv2D(128, 3, 1, 'same', kernel_initializer='he_normal', use_bias=False)(h1)
        # h2 = MaxPooling2D(2, strides=2, padding='valid')(h2)
        h2 = BatchNormalization()(h2)
        h2 = Activation('swish')(h2)
        h3 = Conv2D(192, 3, 1, 'same', kernel_initializer='he_normal', use_bias=False)(h2)
        h3 = BatchNormalization()(h3)
        h3 = Activation('swish')(h3)
        h4 = Conv2D(192, 3, 1, 'same', kernel_initializer='he_normal', use_bias=False)(h3)
        h4 = BatchNormalization()(h4)
        h4 = Activation('swish')(h4)
        h5 = Conv2D(128, 3, 1, 'same', kernel_initializer='he_normal', use_bias=False)(h4)
        h5 = MaxPooling2D(2, strides=2, padding='valid')(h5)
        h5 = BatchNormalization()(h5)
        h5 = Activation('swish')(h5)

        h6 = Flatten()(h5)
        v_output = Dense(1, activation='linear', kernel_initializer='he_normal')(h6)

        model = Model(state_input, v_output)
        model.summary()

        return model, state_input

    ## Design Area
    def design_area(self):
        # np.linspace(시작점, 끝점, 갯수)
        # Design Area의 크기대로 x, y 를 생성 후 합침
        x = np.linspace(0, self.design_length_x, self.action_shape[0])
        y = np.linspace(0, self.design_length_y, self.action_shape[1])
        x, y = np.meshgrid(x, y, indexing='ij')
        design_node = np.stack((x.flatten(), y.flatten()), axis=1)        # shape : ( mesh size , 2 )

        x_net = np.tile(x.flatten(), (self.network_out_shape.size, 1))    # shape : ( self.network_out_shape.size , mesh size )
        x_net_bat = np.tile(x_net, (self.batch_size, 1, 1))               # shape : ( self.batch_size , self.network_out_shape.size , mesh size )
        y_net = np.tile(y.flatten(), (self.network_out_shape.size, 1))    # shape : ( self.network_out_shape.size , mesh size )
        y_net_bat = np.tile(y_net, (self.batch_size, 1, 1))               # shape : ( self.batch_size , self.network_out_shape.size , mesh size )

        design_batch_node = np.stack((x_net_bat, y_net_bat), axis=3)      # shape : ( self.batch_size , self.network_out_shape.size , mesh size, 2)

        return np.float32(design_node), np.float32(design_batch_node)

    # # 네트워크 람다 레이어에 들어갈 값들 계산
    # def coverage_area(self):
    #     net_out_x = self.network_out_shape.shape[0]
    #     net_out_y = self.network_out_shape.shape[1]
    #     coverage_length_x = self.design_length_x / (net_out_x - 1)  # 확률분포 하나당 커버하는 영역설정
    #     coverage_length_y = self.design_length_y / (net_out_y - 1)  # 확률분포 하나당 커버하는 영역설정
    #
    #     indices_x = tf.constant(np.arange(self.batch_size).reshape([-1, 1]))
    #     cover_arr_x = np.arange(net_out_y) * coverage_length_x
    #     cover_arr_x = np.tile(cover_arr_x, (net_out_x, 1)).reshape([1, net_out_x, net_out_y, 1])
    #     cover_arr_x = np.tile(cover_arr_x, (self.batch_size, 1, 1, 1))
    #
    #     indices_y = tf.constant(np.arange(self.batch_size).reshape([-1, 1]))
    #     cover_arr_y = (np.arange(net_out_x) * coverage_length_y).reshape([-1, 1])
    #     cover_arr_y = np.tile(cover_arr_y, (1, net_out_y)).reshape([1, net_out_x, net_out_y, 1])
    #     cover_arr_y = np.tile(cover_arr_y, (self.batch_size, 1, 1, 1))
    #
    #     update_add_x = tf.constant(cover_arr_x, dtype=tf.float32)
    #     update_add_y = tf.constant(cover_arr_y, dtype=tf.float32)
    #
    #     return indices_x, indices_y, update_add_x, update_add_y, coverage_length_x, coverage_length_y


    ## log policy pdf
    @tf.function
    def log_pdf_batch(self, states, action_list):
        tun_prob = self.multivariate_gaussian_mixture_model(states, self.design_batch_node)

        # log_pdf 구하기
        action_prob = tf.where(action_list == 0, tun_prob, 1. - tun_prob)  # 확률 분포에 해당하는 CAD모델 생성 확률
        log_policy_pdf = tf.math.log(action_prob)
        plane_log_probs = tf.reduce_sum(log_policy_pdf, 1, keepdims=True)

        return plane_log_probs

    ## actor policy
    def get_policy_action(self, state):
        # type of action in env is numpy array
        # state shape : shape : (self.unit_num_x, self.unit_num_y, 1) -> (1, self.unit_num_x, self.unit_num_y, 1)
        tun_prob = self.multivariate_gaussian_mixture_model(np.expand_dims(state, axis=0), self.design_node).numpy()

        # 확률 벡터계산 Vectorized Action Probability / https://stackoverflow.com/questions/47722005/vectorizing-numpy-random-choice-for-given-2d-array-of-probabilities-along-an-a참조
        act_prob = tun_prob.reshape([-1, 1])  # Action 확률 [n, 1]으로 리쉐입
        opp_prob = 1 - act_prob  # 1 - Action 확률
        gen_prob = np.concatenate((act_prob, opp_prob), axis=1)  # shape : (n, 2)으로 합침
        r = np.expand_dims(np.random.rand(gen_prob.shape[0]), axis=1)  # gen_prob 의 행 수만큼 0~1 사이 값 생성
        action_list = (gen_prob.cumsum(axis=1) > r).argmax(axis=1)  # gen_prob 의 누적 확률로 decision_list 계산 0.0(unitcell 생성) or 1.0(unitcell 비 생성)

        action_prob = np.where(action_list.reshape([-1, 1]) == 0.0, act_prob, opp_prob)  #  생성이면(0.0 이면) 생성할왁률, 비생성이면(1.0) 생성하지 않을 확률로 매핑
        # # log_pdf 값 추출
        log_old_policy_pdf = np.log(action_prob).sum()

        return log_old_policy_pdf, action_list, tun_prob

    @tf.function
    def multivariate_gaussian_mixture_model(self, states, design_node, sigmoid_factor=3.0):
        mu_x, mu_y, sig_x, sig_y, corr_xy, wt = self.actor(states)  # tensorflow output
        ## 설계영역 index (shape : ( batch size, network size, mesh size, 2 )) 값들을 GMM적용을 위해 전처리
        ## 고차원 연산을 위한 matrix 생성
        # 평균 벡터 생성
        mean_vec = tf.stack([mu_x, mu_y], axis=2)  # shape : ( batch size, network size ) -> ( batch size, network size, 2 )
        mean_vec = tf.expand_dims(mean_vec, axis=-1)  # shape : ( batch size, network size, 2 ) -> ( batch size, network size, 2, 1 )

        # 공분산 행렬 생성
        cov_x = tf.pow(sig_x, 2)
        cov_y = tf.pow(sig_y, 2)
        cov_elem = corr_xy * sig_x * sig_y
        cov_mat_x = tf.stack([cov_x, cov_elem], axis=2)  # shape : ( batch size, network size, 1 ) -> ( batch size, network size, 2 )
        cov_mat_y = tf.stack([cov_elem, cov_y], axis=2)  # shape : ( batch size, network size, 1 ) -> ( batch size, network size, 2 )
        cov_mat = tf.stack([cov_mat_x, cov_mat_y], axis=3)  # shape : ( batch size, network size, 2 ) -> ( batch size, network size, 2, 2 )

        # Tensor Broadcasting 을 위한 차원 확장 / 최종 shape : ( batch size, network size, mesh size, n, n )
        wt = tf.expand_dims(wt, axis=-1)  # shape : ( batch size, network size ) -> ( batch size, network size, 1 )
        mean_vec = tf.expand_dims(mean_vec, axis=2)  # shape : ( batch size, network size, 2, 2 ) -> ( batch size, network size, 1, 2, 1 )
        cov_mat = tf.expand_dims(cov_mat, axis=2)  # shape : ( batch size, network size, 2, 2 ) -> ( batch size, network size, 1, 2, 2 )
        design_node = tf.expand_dims(design_node, axis=-1)  # shape : ( batch size, network size, mesh size, 2 ) -> ( batch size, network size, mesh size, 2, 1 )

        # 다변수 결합 확률 밀도 함수 식 적용 / f = (2*pi)^(-n/2) * (determinant(cov_mat))^(-1/2) * exp{-(1/2)*(design_node-mean_vec)' x cov_mat^(-1) x (design_node-mean_vec)} * 여기서 n은 확률 차원
        subtract_vec = design_node - mean_vec  # shape : ( batch size, network size, mesh size, 2, 1 )
        subtract_vec_T = tf.transpose(subtract_vec, (0, 1, 2, 4, 3))  # shape : ( batch size, network size, mesh size, 1, 2 )
        determinant_cov = tf.linalg.det(cov_mat)  # shape : ( batch size, network size, 1 )
        inv_cov_mat = tf.linalg.inv(cov_mat)  # shape : ( batch size, network size, 1, 2, 2 )

        exponent = tf.squeeze(-0.5 * tf.matmul(tf.matmul(subtract_vec_T, inv_cov_mat), subtract_vec))  # shape : ( batch size, network size, mesh size )
        each_distribution_prob = wt * (tf.pow(np.float32(2. * np.pi), -2 / 2) * tf.pow(determinant_cov, -0.5)) * tf.math.exp(exponent)  # shape : ( batch size, network size, mesh size )
        each_node_prob = tf.reduce_sum(each_distribution_prob, axis=1)  # shape : ( batch size, mesh size )

        ## 다변수 결합 확률 분포 조작
        # 확률 분포 표준화(Standardization)
        each_node_prob_mean = tf.reduce_mean(each_node_prob, axis=1, keepdims=True)
        each_node_prob_std = tf.math.reduce_std(each_node_prob, axis=1, keepdims=True)
        norm_each_node_prob = tf.divide(tf.subtract(each_node_prob, each_node_prob_mean), each_node_prob_std)
        # 가중치(sigmoid_factor=3.0(default)) 부여한 sigmoid 로 확률 중간값 페널티
        tun_prob = tf.pow(1. + tf.exp(-sigmoid_factor * norm_each_node_prob), -1.)


        # # for debug
        # # nan bug search
        # print(f"determinant_cov: {determinant_cov}\n")
        # print(f"inv_cov_mat: {inv_cov_mat}\n")
        # print(f"exponent: {exponent}\n")
        # print(f"each_distribution_prob: {each_distribution_prob}\n")
        # print(f"each_node_prob: {each_node_prob}\n")
        # from tensorflow.python.client import device_lib
        # print(device_lib.list_local_devices())
        # print(f"mean: {mean_vec}")
        # print(f"cov: {cov_mat}")
        # print(f"wt: {wt.numpy().sum()}")
        # print(f"origin_prob_mean: {each_node_prob_mean}")
        # print(f"origin_prob_std: {each_node_prob_std}")
        # print(f"origin_prob_min: {each_node_prob.numpy().min()}")
        # print(f"origin_prob_max: {each_node_prob.numpy().max()}")
        # print(f"manu_prob_min: {tun_prob.numpy().min()}")
        # print(f"manu_prob_max: {tun_prob.numpy().max()}")
        #
        # x = np.linspace(0, 8.0, 80)
        # y = np.linspace(0, 4.0, 40)
        # x, y = np.meshgrid(x, y, indexing='ij')
        # import matplotlib.pyplot as plt
        # fig = plt.figure(figsize=plt.figaspect(0.5))
        # # ---- 1-1 subplot_tfd
        # ax = fig.add_subplot(2, 1, 1, projection='3d')
        # plt.contour(x, y, each_node_prob.numpy().reshape((80, 40)));
        # ax.plot_surface(x, y, each_node_prob.numpy().reshape((80, 40)), cmap='viridis');
        # ax.set_title("origin_prob")
        # # ---- 1-2 subplot_tf
        # ax = fig.add_subplot(2, 1, 2, projection='3d')
        # plt.contour(x, y, tun_prob.numpy().reshape((80, 40)));
        # ax.plot_surface(x, y, tun_prob.numpy().reshape((80, 40)), cmap='viridis');
        # ax.set_title("manuflate_prob")
        # plt.show()

        return tun_prob

    ## actor prediction
    def actor_predict(self, state):
        mu_x, mu_y, sig_x, sig_y, corr_xy, wt = self.actor.predict(state)
        return mu_x, mu_y, sig_x, sig_y, corr_xy, wt

    ## critic prediction
    def critic_predict(self, state):
        return self.critic.predict(state)

    ## train the actor network
    def actor_train(self, log_old_policy_pdfs, states, advantages, action, EPOCH):
        with tf.GradientTape() as tape:
            # current policy pdf
            plane_log_probs = self.log_pdf_batch(states, action)
            # ratio of current and old policies
            ratio = tf.exp(plane_log_probs - log_old_policy_pdfs)
            clipped_ratio = tf.clip_by_value(ratio, 1.0-self.ratio_clipping, 1.0+self.ratio_clipping)
            surrogate = -tf.minimum(ratio * advantages, clipped_ratio * advantages)
            loss = tf.reduce_mean(surrogate)
        dj_dtheta = tape.gradient(loss, self.actor.trainable_variables)
        grads = zip(dj_dtheta, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(grads)
        # #디버그용 텐서보드
        # self.debug_tensorboard(log_old_policy_pdf, plane_log_prob.numpy(), ratio, loss, EPOCH)

    ## train the critic network single gradient update on a single batch data
    def train_on_batch(self, states, td_targets):
        self.critic.train_on_batch(states, td_targets)

    def actor_save_weights(self, path):
        self.actor.save_weights(path)

    ## load actor wieghts
    def actor_load_weights(self, path):
        self.actor.load_weights(path + 'EDPPO_struct_actor.h5')

    ## save critic weights
    def critic_save_weights(self, path):
        self.critic.save_weights(path)

    ## load critic wieghts
    def critic_load_weights(self, path):
        self.critic.load_weights(path + 'EDPPO_struct_critic.h5')