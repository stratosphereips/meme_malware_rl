import os
import random
import sys
import logging
import argparse 

import gym
from gym.envs.registration import register

import numpy as np
from gym import wrappers
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import torch as th

from surrogate import train_surrogate

import malware_rl

logging.basicConfig(filename="training.log",
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)

def init_clean(target, seed):
    # Delete data from previous experiments
    if os.path.exists(os.path.join(save_model_path, 'observations.npy')):
        os.remove(os.path.join(save_model_path, 'observations.npy'))
    if os.path.exists(os.path.join(save_model_path, 'scores.npy')):
        os.remove(os.path.join(save_model_path, 'scores.npy'))
    if os.path.exists(os.path.join(save_model_path, f'lgb_model_{target}_{seed}.txt')):
        os.remove(os.path.join(save_model_path, f'lgb_model_{target}_{seed}.txt'))

    # Delete the memory files too, even though they will be overwritten
    if os.path.exists(os.path.join(data_path, 'observations.npy')):
        os.remove(os.path.join(data_path, 'observations.npy'))
    if os.path.exists(os.path.join(data_path, 'scores.npy')):
        os.remove(os.path.join(data_path, 'scores.npy'))


def register_env(env_name, model_path, target, threshold):
    max_turns = gym.envs.registration.registry.env_specs[f'{target}-train-v0']._kwargs["maxturns"]
    sha256_list = gym.envs.registration.registry.env_specs[f'{target}-train-v0']._kwargs["sha256list"]
    
    if env_name in gym.envs.registration.registry.env_specs:
        logging.debug(f"Remove {env_name} from registry")
        del gym.envs.registration.registry.env_specs[env_name]


    register(
        id=env_name,
        entry_point="malware_rl.envs:LGBEnv",
        kwargs={
            "random_sample": True,
            "maxturns": max_turns,
            "sha256list": sha256_list,
            "save_modified_data": False,
            "model_path": model_path,
            "threshold": threshold
        },
)

def evaluate_agent(agent, env_string, num_episodes, num_queries, outdir, seed=0):
    done = False
    reward = 0
    evasions = 0
    evasion_history = {}
    
    eval_env = gym.make(env_string)
    eval_env = wrappers.Monitor(eval_env, directory=outdir, force=True)
    eval_env.seed(seed)

    # Test the agent in the eval environment
    for i in range(num_episodes):
        ob = eval_env.reset()
        sha256 = eval_env.sha256
        while True:
            action, _ = agent.predict(ob, reward, done)
            ob, reward, done, ep_history = eval_env.step(action)
            if done and reward >= 10.0:
                evasions += 1
                evasion_history[sha256] = ep_history
                break
            elif done:
                break
        if eval_env.queries >= num_queries:
            break 

    logging.debug(f"True episode count in eval: {i+1}")
    logging.debug(f"Skipped binaries: {eval_env.skipped}")

    # Remove the skipped binaries
    total_episodes = (i+1) - eval_env.skipped
    # Output metrics/evaluation stuff
    evasion_rate = (evasions / total_episodes) * 100
    mean_action_count = np.mean(eval_env.get_episode_lengths())
    logging.info(f"{evasion_rate}% samples evaded model.")
    logging.info(f"Average of {mean_action_count} moves to evade model.")
    print("History:", evasion_history)

    queries = eval_env.get_total_steps()
    eval_env.close()

    return queries

argparser = argparse.ArgumentParser()
argparser.add_argument("--target", choices=["ember", "sorel", "sorelFFNN", 'AV1'], default="ember", help="Target to train on")
argparser.add_argument("--seed", type=int, default=26871, help="Random seed")
argparser.add_argument("--num_boosting_rounds", type=int, default=500, help="Number of boosting rounds")
argparser.add_argument("--init_timesteps", type=int, default=256, help="Number of timesteps to train on")
argparser.add_argument("--num_timesteps", type=int, default=2048, help="Number of timesteps to train on")
argparser.add_argument("--eval_timesteps", type=int, default=2048, help="Number of timesteps to evaluate on")
argparser.add_argument("--num_rounds", type=int, default=3, help="Number of rounds to train on")
args = argparser.parse_args()


TARGET = args.target
SEED = args.seed
num_boosting_rounds = args.num_boosting_rounds
init_timesteps = args.init_timesteps
num_timesteps = args.num_timesteps
eval_timesteps = args.eval_timesteps
num_rounds = args.num_rounds

total_queries = 0

random.seed(SEED)
np.random.seed(SEED)

module_path = os.path.split(os.path.abspath(sys.modules[__name__].__file__))[0]
outdir = os.path.join(module_path, "data/logs/ppo-agent-results")
save_model_path = os.path.join(module_path, "malware_rl/envs/utils")
data_path = os.path.join(module_path, f"data/memory/{TARGET}")

# Step 0: Init a policy and gather some data (calculate number of queries)
# First clean all the previous data and models
logging.info(f"Starting experiment with {TARGET} target and seed {SEED}")
init_clean(TARGET, SEED)

# Setting up the environment

target_env = make_vec_env(f"{TARGET}-train-v0", n_envs=1)
target_env.seed(SEED)

# Train the agent
policy_kwargs = dict(activation_fn=th.nn.Tanh,
                     net_arch=dict(pi=[64, 64], vf=[64, 64]))
"""
            gamma=0.9657974584790149,
            n_epochs=10,
            verbose=1, 
            n_steps=128,
            learning_rate=0.0001355978892506237,
            max_grad_norm=0.33249515092054016, 
            tensorboard_log=f"./ppo_{TARGET}_tensorboard/", 
            policy_kwargs=policy_kwargs) 
            # device='cpu')
            """

agent = PPO("MlpPolicy", 
            target_env, 
            gamma=0.854,
            n_epochs=10,
            verbose=1, 
            n_steps=128,
            learning_rate=0.00138,
            max_grad_norm=0.4284, 
            tensorboard_log=f"./ppo_{TARGET}_tensorboard/", 
            policy_kwargs=policy_kwargs) 
            # device='cpu')

# Total timesteps should be a multiple of envs*n_steps 
# agent.learn(total_timesteps=init_timesteps)
# agent.save(f"saved_models/ppo-{TARGET}-train-v0-{SEED}-init")
# target_env.close()

# keep track of the queries
# total_queries += init_timesteps

total_queries = 0

for i in range(num_rounds):

    # Step 1: Run the policy on the target and store queries
    if i == 0:
        logging.info(f"Bootstrap agent learning on the target env. Round: {i+1}")
        agent.learn(total_timesteps=eval_timesteps)
        total_queries += eval_timesteps
    else:
        logging.info(f"Evaluation of the agent on the target env. Round: {i+1}")
        total_queries += evaluate_agent(agent, f"{TARGET}-train-v0", 500, eval_timesteps, outdir, SEED)

    # Step 2: Train a new model (or ensemble) using the new data
    # Step 2a: evaluate model on agreement with the target
    logging.debug(f"Training the surrogate. Round: {i+1}")
    threshold = train_surrogate(TARGET, data_path, save_model_path, SEED)

    # Step 3: use the new model as target and train a new policy
    # Setting up the environment
    register_env('lgb-train-v0', os.path.join(save_model_path, f'lgb_{TARGET}_model_{SEED}.txt'), TARGET, threshold)
    surrogate_env = make_vec_env(f"lgb-train-v0", n_envs=1)
    surrogate_env.seed(SEED)

    # Train the agent
    logging.info(f"Training the agent on the surrogate. Round: {i+1}")

    # Load the saved agent and change environemnt to the surrogate one
    agent.set_env(surrogate_env)
    # These time steps are with the surrogate hence no increase in the counter
    agent.learn(total_timesteps=num_timesteps)
    agent.save(f"saved_models/ppo-model_rl-{TARGET}-train-v0-{SEED}")


logging.info(f"Final eval on the test set. Round: {i+1}")
evaluate_agent(agent, f"{TARGET}-test-v0", 300, 5000, outdir, SEED)

logging.info(f"Total number of queries: {total_queries}")
