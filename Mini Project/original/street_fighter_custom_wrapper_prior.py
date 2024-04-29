
import math
import time
import collections

import gym
import numpy as np

# Custom environment wrapper
class StreetFighterCustomWrapper_prior(gym.Wrapper):
    def __init__(self, env, reset_round=True, rendering=False):
        super(StreetFighterCustomWrapper_prior, self).__init__(env)
        self.env = env

        # Use a deque to store the last 9 frames
        self.num_frames = 9
        self.frame_stack = collections.deque(maxlen=self.num_frames)

        self.num_step_frames = 6

        self.reward_coeff = 3.0

        self.total_timesteps = 0

        self.full_hp = 176
        self.prev_player_health = self.full_hp
        self.prev_oppont_health = self.full_hp

        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(100, 128, 3), dtype=np.uint8)
        
        self.reset_round = reset_round
        self.rendering = rendering
    
    def _stack_observation(self):
        return np.stack([self.frame_stack[i * 3 + 2][:, :, i] for i in range(3)], axis=-1)

    def reset(self):
        observation = self.env.reset()
        
        self.prev_player_health = self.full_hp
        self.prev_oppont_health = self.full_hp

        self.total_timesteps = 0
        
        # Clear the frame stack and add the first observation [num_frames] times
        self.frame_stack.clear()
        for _ in range(self.num_frames):
            self.frame_stack.append(observation[::2, ::2, :])

        return np.stack([self.frame_stack[i * 3 + 2][:, :, i] for i in range(3)], axis=-1)

def step(self, action):
    custom_done = False

    obs, _reward, _done, info = self.env.step(action)
    self.frame_stack.append(obs[::2, ::2, :])

    # Render the game if rendering flag is set to True.
    if self.rendering:
        self.env.render()
        time.sleep(0.01)

    for _ in range(self.num_step_frames - 1):
        obs, _reward, _done, info = self.env.step(action)
        self.frame_stack.append(obs[::2, ::2, :])
        if self.rendering:
            self.env.render()
            time.sleep(0.01)

    curr_player_health = info['agent_hp']
    curr_oppont_health = info['enemy_hp']

    self.total_timesteps += self.num_step_frames

    # Calculate damage dealt and received
    damage_dealt = self.prev_oppont_health - curr_oppont_health
    damage_received = self.prev_player_health - curr_player_health

    # Calculate custom reward
    custom_reward = damage_dealt * 0.1 - damage_received * 0.1

    # Check for end of round
    if curr_player_health <= 0:
        custom_reward -= 100  # large penalty for losing
        custom_done = True
    elif curr_oppont_health <= 0:
        custom_reward += 100  # large reward for winning
        custom_done = True

    # Update health statuses
    self.prev_player_health = curr_player_health
    self.prev_oppont_health = curr_oppont_health

    # Handle the case where the round is not supposed to reset
    if not self.reset_round:
        custom_done = False

    # Normalize the reward
    normalized_reward = 0.001 * custom_reward

    # Max reward is 6 * full_hp = 1054 (damage * 3 + winning_reward * 3) norm_coefficient = 0.001
    return self._stack_observation(), normalized_reward, custom_done, info
