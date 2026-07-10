"""
Q-Learning on FrozenLake from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - init_q_table
import numpy as np

def init_q_table(num_states, num_actions):
    return np.zeros((num_states, num_actions), dtype=np.float64)

# Step 2 - max_q_value
import numpy as np

def max_q_value(q_table, state):
    return float(np.max(q_table[state]))

# Step 3 - greedy_action
import numpy as np

def greedy_action(q_table, state):
    return int(np.argmax(q_table[state]))

# Step 4 - sample_random_action
def sample_random_action(action_space):
    return int(action_space.sample())

# Step 5 - should_explore
def should_explore(epsilon, rng):
    return bool(rng.random() < epsilon)

# Step 6 - epsilon_greedy_action
import numpy as np

def epsilon_greedy_action(q_table, state, epsilon, action_space, rng):
    if should_explore(epsilon, rng):
        return sample_random_action(action_space)
    return greedy_action(q_table, state)

# Step 7 - decay_epsilon
def decay_epsilon(epsilon, decay_rate, min_epsilon):
    return max(min_epsilon, epsilon * decay_rate)

# Step 8 - td_target
def td_target(reward, gamma, q_table, next_state, done):
    if done:
        return float(reward)
    return float(reward) + float(gamma) * max_q_value(q_table, next_state)

# Step 9 - td_error
def td_error(target, q_table, state, action):
    return float(target - q_table[state, action])

# Step 10 - q_learning_update
def q_learning_update(q_table, state, action, reward, next_state, done, alpha, gamma):
    target = td_target(reward, gamma, q_table, next_state, done)
    error = td_error(target, q_table, state, action)
    q_table[state, action] = q_table[state, action] + alpha * error
    return float(q_table[state, action])

# Step 11 - interaction_step
def interaction_step(env, q_table, state, epsilon, alpha, gamma, rng):
    action = epsilon_greedy_action(q_table, state, epsilon, env.action_space, rng)
    next_obs, reward, terminated, truncated, _ = env.step(action)
    done = bool(terminated or truncated)
    next_state = int(next_obs)
    q_learning_update(q_table, state, action, float(reward), next_state, done, alpha, gamma)
    return next_state, float(reward), done

# Step 12 - run_training_episode
def run_training_episode(env, q_table, epsilon, alpha, gamma, rng, max_steps=200):
    obs, _ = env.reset()
    state = int(obs)
    total_reward = 0.0
    for _ in range(max_steps):
        state, reward, done = interaction_step(
            env, q_table, state, epsilon, alpha, gamma, rng
        )
        total_reward += reward
        if done:
            break
    return float(total_reward)

# Step 13 - train_q_learning
def train_q_learning(env, num_episodes, alpha=0.1, gamma=0.99, epsilon_start=1.0, epsilon_min=0.05, epsilon_decay=0.995, seed=0, max_steps=200):
    # train a Q-learning agent for num_episodes; return (q_table, returns)
    rng = np.random.default_rng(seed)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    q_table = init_q_table(num_states, num_actions)
    episode_returns = []
    epsilon = epsilon_start
    for episode in range(num_episodes):
        episode_return = run_training_episode(env, q_table, epsilon, alpha, gamma, rng, max_steps=max_steps)
        episode_returns.append(episode_return)
        epsilon = decay_epsilon(epsilon, epsilon_decay, epsilon_min)
    return (q_table, episode_returns)

# Step 14 - extract_greedy_policy
def extract_greedy_policy(q_table):
    q = np.asarray(q_table)
    return np.argmax(q, axis=1).astype(np.int64)

# Step 15 - run_greedy_episode
def run_greedy_episode(env, policy, seed=None, max_steps=200):
    obs, info = env.reset(seed=seed)
    last_reward = 0.0
    for _ in range(max_steps):
        action = int(policy[obs])
        obs, reward, terminated, truncated, info = env.step(action)
        last_reward = reward
        if terminated or truncated:
            break
    return bool(last_reward > 0)

# Step 16 - evaluate_success_rate
def evaluate_success_rate(env, policy, num_episodes, seed=0, max_steps=200):
    successes = 0
    for i in range(num_episodes):
        if run_greedy_episode(env, policy, seed=seed + i, max_steps=max_steps):
            successes += 1
    return successes / num_episodes

