from algos.qlearning import qlearning
import gym

env = gym.make('LunarLander-v2') #, is_slippery=False)
model = qlearning(env, render="human")

#while True:
#    try:
model.learn()
#    except Exception as e:
#        raise e


# watch trained agent
state = model.env.reset()
rewards = 0

done = False
while True:
    env.reset()
    done = False
    while not done:
        print(f"TRAINED AGENT")

        action = model.choose_action(state)
        new_state, reward, done, info = model.env.step(action)
        env.render()
        state = new_state
        rewards += reward
        print(f"reward {rewards}")
        if rewards > 0:
            x = 4
        print(f"score: {rewards}")

