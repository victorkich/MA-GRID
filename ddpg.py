import pickle
import numpy as np
import torch
import torch.optim as optim
from ddpg_step import ddpg_step
from models.Policy_ddpg import Policy
from models.Value_ddpg import Value
from replay_memory import Memory
from torch.utils.tensorboard import SummaryWriter
from utils.file_util import check_path
from utils.torch_util import device, FLOAT
from utils.zfilter import ZFilter
from environment import MAGRID
from tqdm import tqdm
print("Using:", device)


class DDPG:
    def __init__(self, render=False, num_process=6, memory_size=1000000, lr_p=1e-3, lr_v=1e-3, gamma=0.99, polyak=0.995,
                 explore_size=30000, step_per_iter=10000, batch_size=500, min_update_step=1000, update_step=200,
                 action_noise=0.1, seed=1, model_path=None, env_gamma=0.2, num_agents=3, env_grid=20):
        self.gamma = gamma
        self.polyak = polyak
        self.memory = Memory(memory_size)
        self.explore_size = explore_size
        self.step_per_iter = step_per_iter
        self.render = render
        self.num_process = num_process
        self.lr_p = lr_p
        self.lr_v = lr_v
        self.batch_size = batch_size
        self.min_update_step = min_update_step
        self.update_step = update_step
        self.action_noise = action_noise
        self.model_path = model_path
        self.seed = seed
        self.env_gamma = env_gamma
        self.num_agents = num_agents
        self.env_grid = env_grid
        self._init_model()

    def _init_model(self):
        self.env = MAGRID(self.num_agents, self.env_grid, self.env_gamma)
        self.num_states = self.env.num_states
        self.num_actions = self.env.num_actions

        self.action_low, self.action_high = -1, 1
        # seeding
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        self.env.seed(self.seed)

        self.policy_net = Policy(self.num_states, self.num_actions, self.action_high).to(device)
        self.policy_net_target = Policy(self.num_states, self.num_actions, self.action_high).to(device)

        self.value_net = Value(self.num_states, self.num_actions).to(device)
        self.value_net_target = Value(self.num_states, self.num_actions).to(device)

        self.running_state = ZFilter((self.num_states,), clip=5)

        if self.model_path:
            print("Loading Saved Model {}_ddpg.p".format('MAGRID'))
            self.policy_net, self.value_net, self.running_state = pickle.load(
                open('{}/{}_ddpg.p'.format(self.model_path, 'MAGRID'), "rb"))

        self.policy_net_target.load_state_dict(self.policy_net.state_dict())
        self.value_net_target.load_state_dict(self.value_net.state_dict())

        self.optimizer_p = optim.Adam(self.policy_net.parameters(), lr=self.lr_p)
        self.optimizer_v = optim.Adam(self.value_net.parameters(), lr=self.lr_v)

    def choose_action(self, state, noise_scale):
        """select action"""
        state = FLOAT(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action, log_prob = self.policy_net.get_action_log_prob(state)
        action = action.cpu().numpy()[0]
        # add noise
        noise = noise_scale * np.random.randn(self.num_actions)
        action += noise
        action = np.clip(action, -self.action_high, self.action_high)
        return action 

    def eval(self, i_iter, render=False):
        """evaluate model"""
        state = self.env.reset()
        test_reward = 0
        while True:
            if render:
                self.env.render()
            # state = self.running_state(state)
            action = self.choose_action(state, 0)
            filtered_action = self.filter_action(action)
            state, reward, done = self.env.step(filtered_action)

            test_reward += reward
            if done:
                break
        print(f"Iter: {i_iter}, test Reward: {test_reward}")
        self.env.close()

    def filter_action(self, action):
        for i in range(round(action.size / 3)):
            aux = i * 3
            if abs(action[aux]) > abs(action[aux + 1]) and abs(action[aux]) > abs(action[aux + 2]):
                action[aux] = -1 if action[aux] < 0 else 1
                action[aux + 1] = 0
                action[aux + 2] = 0
            elif abs(action[aux]) < abs(action[aux + 1]) and abs(action[aux + 1]) > abs(action[aux + 2]):
                action[aux] = 0
                action[aux + 1] = -1 if action[aux + 1] < 0 else 1
                action[aux + 2] = 0
            else:
                action[aux] = 0
                action[aux + 1] = 0
                action[aux + 2] = -1 if action[aux + 2] < 0 else 1
        return action.astype(np.int32)

    def learn(self, writer, i_iter):
        """interact"""
        global_steps = (i_iter - 1) * self.step_per_iter
        log = dict()
        num_steps = 0
        num_episodes = 0
        total_reward = 0.
        min_episode_reward = float('inf')
        max_episode_reward = float('-inf')

        for _ in tqdm(range(self.step_per_iter)):
            state = self.env.reset()
            episode_reward = 0
            for t in range(10000):
                if self.render:
                    self.env.render()

                if global_steps < self.explore_size:  # explore
                    action = self.env.get_action_space_sample()
                    filtered_action = action
                else:  # action with noise
                    action = self.choose_action(state, self.action_noise)
                    filtered_action = self.filter_action(action)

                next_state, reward, done = self.env.step(filtered_action)
                mask = 0 if done else 1
                # ('state', 'action', 'reward', 'next_state', 'mask', 'log_prob')
                self.memory.push(state, action, reward, next_state, mask, None)

                episode_reward += reward
                global_steps += 1
                num_steps += 1

                if global_steps >= self.min_update_step and global_steps % self.update_step == 0:
                    for _ in range(self.update_step):
                        batch = self.memory.sample(
                            self.batch_size)  # random sample batch
                        self.update(batch)

                if done or num_steps >= self.step_per_iter:
                    break

                state = next_state

            num_episodes += 1
            total_reward += episode_reward
            min_episode_reward = min(episode_reward, min_episode_reward)
            max_episode_reward = max(episode_reward, max_episode_reward)

        # self.env.close()
        log['num_steps'] = num_steps
        log['num_episodes'] = num_episodes
        log['total_reward'] = total_reward
        log['avg_reward'] = total_reward / num_episodes
        log['max_episode_reward'] = max_episode_reward
        log['min_episode_reward'] = min_episode_reward

        print(f"Iter: {i_iter}, num steps: {log['num_steps']}, total reward: {log['total_reward']: .4f}, "
              f"min reward: {log['min_episode_reward']: .4f}, max reward: {log['max_episode_reward']: .4f}, "
              f"average reward: {log['avg_reward']: .4f}")

        # record reward information
        writer.add_scalar("total reward", log['total_reward'], i_iter)
        writer.add_scalar("average reward", log['avg_reward'], i_iter)
        writer.add_scalar("min reward", log['min_episode_reward'], i_iter)
        writer.add_scalar("max reward", log['max_episode_reward'], i_iter)
        writer.add_scalar("num steps", log['num_steps'], i_iter)

    def update(self, batch):
        """learn model"""
        batch_state = FLOAT(batch.state).to(device)
        batch_action = FLOAT(batch.action).to(device)
        batch_reward = FLOAT(batch.reward).to(device)
        batch_next_state = FLOAT(batch.next_state).to(device)
        batch_mask = FLOAT(batch.mask).to(device)

        # update by DDPG
        alg_step_stats = ddpg_step(self.policy_net, self.policy_net_target, self.value_net, self.value_net_target,
                                   self.optimizer_p, self.optimizer_v, batch_state, batch_action, batch_reward,
                                   batch_next_state, batch_mask, self.gamma, self.polyak)

    def save(self, save_path):
        """save model"""
        check_path(save_path)
        pickle.dump((self.policy_net, self.value_net, self.running_state), open('{}/{}_ddpg.p'.format(save_path, 'MAGRID'), 'wb'))


if __name__ == '__main__':
    log_path = "../log/"
    seed = 1
    base_dir = log_path + "/DDPG_exp{}".format(seed)
    writer = SummaryWriter(base_dir)
    max_iter = 500
    model_path = 'trained_models'
    save_iter = 50
    render = False
    eval_iter = 50
    ddpg = DDPG()

    for i_iter in range(1, max_iter + 1):
        ddpg.learn(writer, i_iter)
        if i_iter % eval_iter == 0:
            ddpg.eval(i_iter, render=render)
        if i_iter % save_iter == 0:
            ddpg.save(model_path)
        torch.cuda.empty_cache()
