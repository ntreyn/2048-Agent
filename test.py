from env2048 import env_2048


def main():
    env = env_2048()

    am = {'l': 0, 'r': 1, 'u': 2, 'd': 3}
    done = False
    
    while not done:
        # env.render()
        # str_action = input("Choose action: (l, r, u, d) ")
        # action = am[str_action]
        action = env.sample_action()
        next_state, reward, done, _ = env.step(action)
        env.render()
        print(done)
    env.render()


if __name__ == "__main__":
    main()