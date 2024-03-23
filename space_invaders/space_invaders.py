import gym
import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
import cv2

# Set random seed for reproducibility
random.seed(0)

def main():
    # Create the SpaceInvaders environment
    env = gym.make('SpaceInvaders-v0')
    env.seed(0)
    
    # Set hyperparameters
    epsilon = 0.1  # Exploration rate
    gamma = 0.99   # Discount factor

    # Define the neural network model
    model = keras.Sequential([
        keras.layers.Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=(84, 84, 4)),
        keras.layers.Conv2D(64, (4, 4), strides=(2, 2), activation='relu'),
        keras.layers.Conv2D(64, (2, 2), activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(6, activation='linear'),  # 6 actions in SpaceInvaders
    ])

    # Compile the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    num_episodes = 100
    for i in range(num_episodes):
        episode_reward = 0
        state = preprocess_observation(env.reset())  # Preprocess the initial observation
        while True:
            # Epsilon-greedy policy for action selection
            if np.random.rand() < epsilon:
                action = env.action_space.sample()  # Random action with epsilon probability
            else:
                action = np.argmax(model.predict(np.expand_dims(state, axis=0)))  # Choose action with highest Q-value

            # Take a step in the environment
            next_state, reward, done, _ = env.step(action)
            next_state = preprocess_observation(next_state)  # Preprocess the next observation

            # Update experience buffer
            episode_reward += reward
            experience_buffer.append((state, action, reward, next_state, done))

            state = next_state

            if done:
                print("Reward of episode {}: {}".format(i + 1, episode_reward))
                break

    # Train the model using experience replay
    minibatch = random.sample(experience_buffer, batch_size)
    update_dqn(model, minibatch, gamma)

def preprocess_observation(observation):
    # Preprocess the observation by converting to grayscale and resizing
    return cv2.resize(cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY), (84, 84))

def update_dqn(model, minibatch, gamma):
    states, targets = [], []
    for state, action, reward, next_state, done in minibatch:
        # Predict Q-values for the current state
        q_values = model.predict(state[np.newaxis])
        # Predict Q-values for the next state
        next_q_values = model.predict(next_state[np.newaxis])
        # Update Q-value for the chosen action based on Bellman equation
        q_values[0][action] = reward if done else reward + gamma * np.max(next_q_values)

        states.append(state)
        targets.append(q_values)

    # Convert lists to numpy arrays
    states, targets = np.vstack(states), np.vstack(targets)
    # Update the model weights using the states and target Q-values
    model.fit(states, targets, epochs=1, verbose=0)

if __name__ == "__main__":
    main()
