from tensorforce.agents import Agent
from tensorforce import Runner
from agentenvironment import AgentEnvironment
from tensorboard import program
from tensorboard import program
import matplotlib.pyplot as plt



def trainRLModel(num_training_episodes, save_model):
    tracking_address = "summaries/run"

    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', tracking_address])
    url = tb.launch()
    print(f"Tensorflow listening on {url}")

    environment = AgentEnvironment(1)

    agent = Agent.create(
        agent='a2c',
        environment = environment,
        network = [
            dict(type='dense', size=128, activation='relu'),
            dict(type='dense', size=128, activation='relu')
        ],
        # A2C optimization parameters
        batch_size=100, update_frequency=100, learning_rate=1e-2, horizon=10,
        # Reward estimation
        discount=0.99, predict_terminal_values=False,
        reward_processing=None,
        #Critic network and optimizer
        critic = [
            dict(type='dense', size=128, activation='relu'),
            dict(type='dense', size=128, activation='relu')
        ],
        critic_optimizer=dict(optimizer='adam', learning_rate=1e-2),
        # Regularization
        l2_regularization=0.0, entropy_regularization=0.0,
        # Preprocessing
        state_preprocessing='linear_normalization',
        # Exploration
        exploration=0.0, variable_noise=0.0,
        # Default additional config values
        config=None,
        # Save agent every 10 updates and keep the 5 most recent checkpoints
        saver=dict(directory='model', frequency=10, max_checkpoints=5),
        # Log all available Tensorboard summaries
        summarizer=dict(directory='summaries', filename = 'run', summaries='all'),
        # Do not record agent-environment interaction trace
        recorder=None
    )

    # Initialize the runner
    runner = Runner(agent=agent, environment=environment, max_episode_timesteps=200)

    # Train for 500 episodes
    runner.run(num_episodes=num_training_episodes)
    runner.close()

    if save_model:
        agent.save(directory = 'saved_models', format = 'numpy')
    
    return agent, environment

def runModel(agent, environment, max_time_steps):
    state = environment.reset()
    done = 0
    total_reward = 0
    while not done:
        omega  = agent.act(state, independent = True)
        next_state, done, reward = environment.execute(omega)
        print(done)
        total_reward += reward
        state = next_state
        environment.plot()
        plt.pause(0.1)
    
    print(total_reward)


if __name__ == '__main__':
    num_training_episodes = 100
    agent, environment = trainRLModel(num_training_episodes, True)
    max_time_steps = 200
    runModel(agent, environment, max_time_steps)
