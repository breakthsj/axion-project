# EDPPO main

from EDPPO_env_v1 import SimLabEnv
from EDPPO_agent_v1 import EDPPOagent

def main():
    max_episode_num = 100000
    parallel_num = 1
    env = SimLabEnv()

    agent = EDPPOagent(env)
    agent.train(max_episode_num, parallel_num)

    # agent.plot_save()


if __name__ == "__main__":
    main()