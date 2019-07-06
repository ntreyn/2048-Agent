from env2048 import env_2048
from dqn import DQN
from parameters import core_argparser, extra_params

import argparse

def train_dqn(env, args):

    agent = DQN(env, args)
    agent.train()
    
    total_episodes = args.episodes
    max_steps = 10

    for episode in range(total_episodes):
        print(episode, agent.epsilon, end='\r')

        state = env.reset()
        done = False

        for step in range(max_steps):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)

            agent.push(state, action, reward, next_state, done)
            agent.learn(episode)

            state = next_state

            if done:
                break
        
        if episode % 5 == 0:
            max_steps += 10

    return agent

def eval_agent(env, agent, eval_iter=3):
    agent.eval() 
    total_reward = 0.0

    for i_episode in range(eval_iter):
        state = env.reset()
        episode_reward = 0.0

        while True:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            if done:
                break
            state = next_state
        total_reward += episode_reward
    
    env.render()

    average_reward = total_reward / (1.0 * eval_iter)
    return average_reward



def human_play(env):
    am = {'l': 0, 'r': 1, 'u': 2, 'd': 3}
    done = False
    
    while not done:
        env.render()
        str_action = input("Choose action: (l, r, u, d) ")
        action = am[str_action]
        next_state, reward, done, _ = env.step(action)
    env.render()

def main(args):
    env = env_2048()
    dqn_agent = train_dqn(env, args)
    eval_agent(env, dqn_agent)


if __name__ == "__main__":
    ARGPARSER = argparse.ArgumentParser(parents=[core_argparser()])
    PARAMS = extra_params(ARGPARSER.parse_args())
    main(PARAMS)