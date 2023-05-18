import pickle
import gym
import random
import typing
import numpy as np
from numpy.random import default_rng
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import DummyVecEnv

class qlearning(object):
    def __init__(self, env, render=None, callbacks=None, saves=None):
        self.callbacks = callbacks
        self.render = render
        if isinstance(env, DummyVecEnv):
            self.env = env.envs[0]
        else:
            self.env = env
        self.x = 4
        self.qtable = {}
        self.actions_by_state = {}
        self.max_steps = 1000000

        # hyperparameters
        self.learning_rate = 0.1
        self.discount_rate = 0.1
        self.epsilon = .1
        #self.decay_rate = 0.001
        if saves is not None:
            self.qtable = saves[0]
            self.actions_by_state = saves[1]

    def check_epsilon(self):
        if np.random.normal(0, 1) < self.epsilon:
            return True
        return False
    def get_random_action(self):
        rng = default_rng()
        action_space = self.env.action_space
        shape = action_space.shape

        if isinstance(action_space, gym.spaces.Discrete):
            action = rng.integers(action_space.n)
            return action
        else:
            return np.random.randint(action_space.nvec)

    def get_hashable_state(self, state):
        if isinstance(state, list):
            if isinstance(state[0], list):
                return str(state)
            else:
                return tuple(state)
        if isinstance(state, typing.Hashable):
            return state

        if isinstance(state, np.ndarray):
            state = state.reshape(np.shape(self.env.observation_space))
            return state.tobytes()
        return state
    def get_max_qvalue_by_state(self, state):
        if isinstance(state, np.ndarray):
            state = state.tobytes()
        actions_by_state = self.actions_by_state.get(state, [])
        q_value = 0
        #random_action = self.get_random_action()
        best_action = None
        for action in actions_by_state:
            temp_action = action # self.get_hashable_state(action)
            state_action_bytes = np.array((state, temp_action)).tobytes()
            temp_value = self.qtable.get(state_action_bytes, 0)
            if temp_value > q_value:
                q_value = temp_value
                best_action = temp_action
                #print(f"q: {q_value}")
        return q_value

    def choose_action(self, state):
        #state_bytes = self.get_hashable_state(state)
        actions_by_state = self.actions_by_state.get(str(state), [])
        q_value = 0
        #random_action = self.get_random_action()
        best_action = None
        for action in actions_by_state:
            temp_action = action # self.get_hashable_state(action)
            state_action_bytes = np.array((str(state), temp_action)).tobytes()
            temp_value = self.qtable.get(str(state), 0)
            if temp_value > q_value:
                q_value = temp_value
                best_action = temp_action
                print(f"q: {q_value}")
        if best_action is None or self.check_epsilon is True:
            best_action = self.get_random_action()
        return best_action
    def learn(self):
        state = self.env.reset()
        for x in range(1000000):
            with open('qtable', 'wb') as fp:
                pickle.dump(self.qtable, fp)
            with open('actions_by_state', 'wb') as fp:
                pickle.dump(self.actions_by_state, fp)
            for i in range(self.max_steps):

                action = self.choose_action(str(state))
                temp_action = action #self.get_hashable_state(action)
                new_state, reward, done, info = self.env.step(action)
                last_state = state
                #state = new_state.reshape(np.shape(self.env.observation_space))
                state = new_state
                state_action_bytes = np.array((str(state), action))
                q_value = self.qtable.get(str(state_action_bytes), 0)
                self.qtable[str(state_action_bytes)] = q_value + self.learning_rate * (reward + self.discount_rate * self.get_max_qvalue_by_state(str(state)) - q_value)

                state_actions = self.actions_by_state.get(str(state), [])
                print(f"action/reward:{(action, reward)}")
                print(f"num state_actions/q_value: {len(state_actions)}/{q_value}")
                if self.render is not None:
                    self.env.render()
                try:
                    action_space = self.env.action_space
                    if isinstance(action_space, gym.spaces.Discrete):
                        if action not in state_actions:
                            state_actions.append(action)
                    else:
                        if len(state_actions) > 0:
                            if not any(action in state_action for state_action in state_actions):
                                state_actions.append(action)
                        else:
                            state_actions.append(action)
                    self.actions_by_state[str(state)] = state_actions

                    if done is True:
                        state = self.env.reset()
                        done = False
                        break

                except Exception as e:
                    raise e


