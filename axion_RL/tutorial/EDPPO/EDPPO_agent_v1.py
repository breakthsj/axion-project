# EDPPO Agent

import numpy as np
import matplotlib.pyplot as plt
import time

import os
import datetime
import shutil
from pathlib import Path

import psutil
import gc

from EDPPO_network_v1 import Network
import ray


class EDPPOagent(object):
    def __init__(self, env):
        ray.init(num_cpus=24)
        # hyperparameters
        self.GAMMA = 0.95       # 현재는 안쓰임
        self.GAE_LAMBDA = 0.9   # 현재는 안쓰임
        self.BATCH_SIZE = 40
        self.ACTOR_LEARNING_RATE = 0.000001
        self.CRITIC_LEARNING_RATE = 0.00001
        self.RATIO_CLIPPING = 0.2
        self.EPOCHS = 150
        self.opt_value = np.array(100)  # 초기 최적값 | 의미는 없고 compliance값에서 나올 수 없는 높은값

        # initial value
        self.episode = 0
        self.date = datetime.datetime.now().strftime("%Y_%m%d_%H")
        self.plot_dir = f"./Plane_Prob_plot/{self.date}_plot"
        self.action_dir = './save_action/'
        self.optimal_dir = './FEniCS/optimal_' + self.date

        self.env = env
        # get state dimension
        self.state_shape = env.observation_space.shape
        # get action dimension
        self.action_shape = env.action_space.shape
        # # get design length
        self.design_length_x, self.design_length_y = env.design_length_x, env.design_length_y

        # create actor and critic networks
        self.network = Network.remote(self.state_shape, self.action_shape, self.design_length_x, self.design_length_y, self.ACTOR_LEARNING_RATE,
                                      self.CRITIC_LEARNING_RATE, self.RATIO_CLIPPING, self.BATCH_SIZE)

        # save the results
        self.save_epi_reward = []
        self.states, self.actions, self.rewards, self.log_old_policy_pdfs = [], [], [], []
        self.gaes_samples, self.y_i_samples = [], []


    def gae_target_ep(self, rewards, v_values):
        n_step_targets = np.zeros_like(rewards)
        gae = np.zeros_like(rewards)
        gae_cumulative = 0
        forward_val = 0

        for k in reversed(range(0, len(rewards))):
            delta = rewards[k] + self.GAMMA * forward_val - v_values[k]
            gae_cumulative = self.GAMMA * self.GAE_LAMBDA * gae_cumulative + delta
            gae[k] = gae_cumulative
            forward_val = v_values[k]
            n_step_targets[k] = gae[k] + v_values[k]
        return gae, n_step_targets

    ## convert (list of np.array) to np.array
    def unpack_batch(self, batch):
        unpack = batch[0]
        for idx in range(len(batch)-1):
            unpack = np.append(unpack, batch[idx+1], axis=0)

        return unpack


    @ray.remote
    def multi_batch_ep(self, cpu_id, opt_val):
        # initialize batch
        batch_state, batch_action, batch_reward = [], [], []
        batch_log_old_policy_pdf = []

        # reset episode
        episode_reward, done = 0, False
        step = 0
        # reset the environment and observe the first state
        state = self.env.reset(cpu_id)

        while not done:
            # 메모리 점유율이 70% 넘어가면 garbage collector 실행
            # if psutil.virtual_memory().percent >= 70.0:
            # gc.collect()

            step += 1
            actions_object_id = self.network.get_policy_action.remote(state)
            log_old_policy_pdf, action_list, tun_prob = ray.get(actions_object_id)
            next_state, reward, done, target_val, opt_val, vol, spend_time = self.env.step(action_list, opt_val, cpu_id)
            # change shape (state_dim,) -> (1, state_dim), same to action, reward, log_old_policy_pdf
            state = np.expand_dims(state, axis=0)
            action = np.expand_dims(action_list, axis=0)
            reward = np.expand_dims(reward, axis=0)
            log_old_policy_pdf = np.expand_dims(log_old_policy_pdf, axis=0)

            # append to the batch
            batch_state.append(state)
            batch_action.append(action)
            batch_reward.append(reward)
            batch_log_old_policy_pdf.append(log_old_policy_pdf)

            state = next_state
            episode_reward += reward[0]

        # compute gae and TD targets
        unpack_state = self.unpack_batch(batch_state)
        unpack_reward = self.unpack_batch(batch_reward)

        v_values_object_id = self.network.critic_predict.remote(unpack_state)
        v_values = ray.get(v_values_object_id)
        gaes_sample, y_i_sample = self.gae_target_ep(unpack_reward, v_values)

        return batch_state, batch_action, batch_reward, batch_log_old_policy_pdf, episode_reward, opt_val, target_val, step, cpu_id, vol, gaes_sample, y_i_sample, tun_prob, spend_time


    def train(self,max_episode_num, parallel_num):
        # 프로세스 지정용
        proc_ids = {x for x in range(parallel_num)}
        error = 0

        # 병렬처리 시작
        result_ids = [EDPPOagent.multi_batch_ep.remote(self, proc_ids.pop(), self.opt_value) for _ in range(parallel_num)]
        while self.episode != max_episode_num:
            # 설정된 최대 에피소드에 도달하면 중지
            # 병렬처리 데이터 받아오기
            done_id, result_ids = ray.wait(result_ids)

            # 병렬 처리 중 오류 발생하면 해당 프로세스의 에피소드 오류처리하고 프로세스 재배치
            try:
                batch_state, batch_action, batch_reward, batch_log_old_policy_pdf, episode_reward, opt_val, target_val, step, cpu_id, vol, gaes_sample, y_i_sample, tun_prob, spend_time = ray.get(done_id[0])
                proc_ids.discard(cpu_id)

            except Exception as ex:
                print("프로세스 오류 발생", ex)
                error += 1
                proc_ids = {x for x in range(parallel_num)}
                continue


            finally:
                if (len(proc_ids) == error) & (error != 0):
                    for _ in range(error):
                        result_ids.append(EDPPOagent.multi_batch_ep.remote(self, proc_ids.pop(), self.opt_value))
                        error -= 1
            self.episode += 1

            # 텐서보드 기록
            check_state = np.ones((self.state_shape[0], self.state_shape[1], 1))
            actor_output_id = self.network.actor_predict.remote(np.expand_dims(check_state, axis=0))
            mu_x, mu_y, sig_x, sig_y, corr_xy, wt = ray.get(actor_output_id)
            tensorB_object_id = self.network.draw_Tensorboard.remote(episode_reward, step, self.episode, batch_action, batch_log_old_policy_pdf, gaes_sample, y_i_sample,mu_x, mu_y, sig_x, sig_y, wt)
            ray.get(tensorB_object_id)

            # plot 용 acion 저장
            Path(self.action_dir).mkdir(parents=True, exist_ok=True)
            np.savetxt(self.action_dir + 'action_plane.txt', tun_prob)
            if self.episode == 1:
                # initial plot
                Path(self.plot_dir).mkdir(parents=True, exist_ok=True)
                self.plot_save(self.plot_dir)

            # 찾은 최적 결과 파일 저장
            if self.opt_value > opt_val:
                Path(self.optimal_dir).mkdir(parents=True, exist_ok=True)
                self.env.set_dir(cpu_id)
                result_dir = self.env.result_dir
                shutil.copy2(result_dir + '.xdmf', os.path.join(self.optimal_dir, f"Ep_{self.episode} Reward_{episode_reward:.3f} Target_{opt_val :.5f} Area_{vol:.3f}.xdmf"))
                shutil.copy2(result_dir + '.h5', os.path.join(self.optimal_dir, f"Ep_{self.episode} Reward_{episode_reward:.3f} Target_{opt_val :.5f} Area_{vol:.3f}.h5"))
                self.plot_save(self.optimal_dir)

            # 최소 opt_value값 저장
            self.opt_value = np.where(self.opt_value > opt_val, opt_val, self.opt_value)

            # 멀티 에피소드 저글링
            result_ids.append(EDPPOagent.multi_batch_ep.remote(self, cpu_id, self.opt_value))

            # 현재 상태 출력
            print(f'Episode: {self.episode} | Respectively Episode Time taken: {spend_time:.3f} | Compliance: {target_val:.5f} |'
                  f' Opt_Compliance: {self.opt_value:.5f} | Area: {vol:.5f} | Reward: [{episode_reward:.5f}]')

            # extract batched states, actions, td_targets, advantages
            states = self.unpack_batch(batch_state)
            actions = self.unpack_batch(batch_action)
            rewards = self.unpack_batch(batch_reward)
            log_old_policy_pdfs = self.unpack_batch(batch_log_old_policy_pdf)

            # 데이터 중복 방지
            if len(self.states) == 0:
                self.states = states
                self.actions = actions
                # self.rewards = rewards
                self.log_old_policy_pdfs = log_old_policy_pdfs
                self.gaes_samples = gaes_sample
                self.y_i_samples = y_i_sample
            else:
                self.states = np.vstack((self.states, states))
                self.actions = np.vstack((self.actions, actions))
                # self.rewards = np.vstack((self.rewards, rewards))
                self.log_old_policy_pdfs = np.vstack((self.log_old_policy_pdfs, log_old_policy_pdfs))
                self.gaes_samples = np.vstack((self.gaes_samples, gaes_sample))
                self.y_i_samples = np.vstack((self.y_i_samples, y_i_sample))

            self.save_epi_reward.append(episode_reward)

            # continue until batch becomes full
            if len(self.states) >= self.BATCH_SIZE:
                network_start_time = time.time()

                # for EPOCH in range(self.EPOCHS):
                #     # train
                #     actor_train_object_id = self.network.actor_train.remote(self.log_old_policy_pdfs, self.states, self.gaes_samples, self.actions, EPOCH)
                #     ray.get(actor_train_object_id)
                #
                #     critic_train_object_id = self.network.train_on_batch.remote(self.states, self.y_i_samples)
                #     ray.get(critic_train_object_id)

                # update the networks
                try:
                    for EPOCH in range(self.EPOCHS):
                        # train
                        actor_train_object_id = self.network.actor_train.remote(self.log_old_policy_pdfs, self.states, self.gaes_samples, self.actions, EPOCH)
                        ray.get(actor_train_object_id)

                        critic_train_object_id = self.network.train_on_batch.remote(self.states, self.y_i_samples)
                        ray.get(critic_train_object_id)
                except:
                    print("network 오류 발생")
                    root_logdir = os.path.join(os.curdir, "Learning_Error_log")
                    sub_dir_name = datetime.datetime.now().strftime(f"%Y%m%d_%H%M_{self.episode}")
                    error_dir = os.path.join(root_logdir, sub_dir_name)
                    os.makedirs(error_dir)

                    time.sleep(10)
                    try:
                        for _ in range(self.EPOCHS):
                            # train
                            actor_train_object_id = self.network.actor_train.remote(self.log_old_policy_pdfs, self.states, self.gaes_samples, self.actions)
                            ray.get(actor_train_object_id)

                            critic_train_object_id = self.network.train_on_batch.remote(self.states, self.y_i_samples)
                            ray.get(critic_train_object_id)
                    except:
                        pass

                finally:
                    print(f"Total Network update elapsed time: {time.time() - network_start_time:.3f}")

                # 배치 비움
                self.states = []
                self.actions = []
                # self.rewards = []
                self.log_old_policy_pdfs = []
                self.gaes_samples = []
                self.y_i_samples = []

            ## save weights every episode
            if self.episode % 10 == 0:
                actor_save_object_id = self.network.actor_save_weights.remote("./save_weights/EDPPO_struct_actor.h5")
                ray.get(actor_save_object_id)

                critic_save_object_id = self.network.critic_save_weights.remote("./save_weights/EDPPO_struct_critic.h5")
                ray.get(critic_save_object_id)
                np.savetxt('./save_weights/ep_reward.txt', self.save_epi_reward)

                if self.episode % 100 == 0:  # 100ep 마다 확률분포 plot 저장
                    self.plot_save(self.plot_dir)

    ## plot_plane
    def plot_save(self, dir):
        x = np.linspace(0, self.design_length_x, self.state_shape[0])
        y = np.linspace(0, self.design_length_y, self.state_shape[1])
        x, y = np.meshgrid(x, y, indexing="ij")

        load_prob = np.loadtxt('./save_action/action_plane.txt')

        ax = plt.axes(projection='3d')
        plt.contour(x, y, load_prob.reshape((self.state_shape[0], self.state_shape[1])));
        ax.plot_surface(x, y, load_prob.reshape((self.state_shape[0], self.state_shape[1])), cmap='viridis');
        plt.savefig(dir + f"/ep_{self.episode}_plot")

from EDPPO_env_v1 import SimLabEnv

def main():
    max_episode_num = 1000000
    parallel_num = 20
    env = SimLabEnv()

    agent = EDPPOagent(env)
    agent.train(max_episode_num, parallel_num)

    # agent.plot_save()


if __name__ == "__main__":
    main()