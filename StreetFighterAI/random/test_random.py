

import os
import time 

import retro
from stable_baselines3 import PPO

from street_fighter_custom_wrapper_random import StreetFighterCustomWrapper


RESET_ROUND = True
# Whether to reset the round when fight is over. 
RENDERING = True    # Whether to render the game screen.

base = r'ppo_ryu_base_3000000_steps'
Diambra = r"ppo_ryu_basic_3000000_steps"
adanvanced = r"ppo_ryu_both_3000000_steps"
king = r"ppo_ryu_king_10000000_steps"
random = r"ppo_ryu_gerneral_9000000_steps"
MODEL_NAME = base
MODEL_NAME = Diambra
#RESET_ROUND = False
MODEL_NAME = king
MODEL_NAME = random
RANDOM_ACTION = False
NUM_EPISODES = 100 # Make sure NUM_EPISODES >= 3 if you set RESET_ROUND to False to see the whole final stage game.
MODEL_DIR = r"StreetFighterAI/train_result/"

def make_env(game, state):
    def _init():
        env = retro.make(
            game=game, 
            state=state, 
            use_restricted_actions=retro.Actions.FILTERED,
            obs_type=retro.Observations.IMAGE
        )
        env = StreetFighterCustomWrapper(env, reset_round=RESET_ROUND, rendering=RENDERING)
        return env
    return _init

game = "StreetFighterIISpecialChampionEdition-Genesis"
env = make_env(game, state="Champion.Level12.RyuVsBison")()
# model = PPO("CnnPolicy", env)

if not RANDOM_ACTION:
    model = PPO.load(os.path.join(MODEL_DIR, MODEL_NAME), env=env)

obs = env.reset()
done = False

num_episodes = NUM_EPISODES
episode_reward_sum = 0
num_victory_all = 0


print("\nFighting Begins!\n")

for _ in range(num_episodes):
    done = False

    total_reward = 0

    while not done:
        timestamp = time.time()

        if RANDOM_ACTION:
            obs, reward, done, info = env.step(env.action_space.sample())
        else:
            action, _states = model.predict(obs)
            obs, reward, done, info = env.step(action)

        if reward != 0:
            total_reward += reward
            print("Reward: {:.3f}, playerHP: {}, enemyHP:{}".format(reward, info['agent_hp'], info['enemy_hp']))

    if info['enemy_hp'] < 0:
        print("Victory!")
        num_victory_all +=1
    print("Total reward: {}\n".format(total_reward))
    episode_reward_sum += total_reward
    obs = env.reset()


env.close()
print("Winning rate: {}".format(1.0 * num_victory_all / num_episodes))
if RANDOM_ACTION:
    print("Average reward for random action: {}".format(episode_reward_sum/num_episodes))
else:
    print("Average reward for {}: {}".format(MODEL_NAME, episode_reward_sum/num_episodes))