# Deep Q-Learning Stock Trading Agent Using Agentic AI

This tutorial will walk you through building a simple stock trading agent using Deep Q-Learning (DQN) with PyTorch, pandas, and yfinance. We’ll explain every step, from data collection to agent training and evaluation, so you can understand the logic and implementation behind each part.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Setup: Installing and Importing Libraries](#setup)
3. [Data Collection](#data-collection)
4. [Feature Engineering](#feature-engineering)
5. [State Representation](#state-representation)
6. [Trading Environment](#trading-environment)
7. [Deep Q-Network (DQN) Model](#deep-q-network-dqn-model)
8. [DQN Agent](#dqn-agent)
9. [Training the Agent](#training-the-agent)
10. [Testing the Trained Agent](#testing-the-trained-agent)
11. [Summary](#summary)

---

## Introduction

**Goal:**
Build an AI agent that learns to trade a stock (e.g., Apple: `AAPL`) using historical data and reinforcement learning. The agent will decide when to buy, sell, or hold based on recent price movements and technical indicators.

---

## Setup

### 1. Install Required Libraries

```python
!pip install yfinance
```


### 2. Import Libraries

```python
import yfinance as yf           # For downloading stock data
import pandas as pd             # For data manipulation
import numpy as np              # For numerical operations
import torch                    # For deep learning
import torch.nn as nn           # For neural network layers
import torch.optim as optim     # For optimization algorithms
import random                   # For random sampling
from collections import deque   # For experience replay buffer
```


---

## Data Collection

### Download Historical Stock Data

```python
symbol = "AAPL"
start_date = "2020-01-01"
end_date = "2025-02-14"

data = yf.download(symbol, start=start_date, end=end_date)
display(data.head())
display(data.shape)
display(data.info())
```

- **What’s happening?**
    - We download daily price data for Apple (AAPL) from Yahoo Finance.
    - The `data` DataFrame contains columns like `Open`, `High`, `Low`, `Close`, `Adj Close`, and `Volume`.

---

## Feature Engineering

### Add Technical Indicators

```python
data["SMA_5"] = data["Close"].rolling(window=5).mean()    # 5-day Simple Moving Average
data["SMA_20"] = data["Close"].rolling(window=20).mean()  # 20-day Simple Moving Average
data["Returns"] = data["Close"].pct_change()              # Daily returns

display(data.head())
display(data.shape)
display(data.info())
```

- **Why?**
    - Moving averages smooth out price data and help the agent spot trends.
    - Returns show daily price changes, which can signal volatility or momentum.


### Clean the Data

```python
data.dropna(inplace=True)             # Remove rows with missing values (from moving averages)
data.reset_index(drop=True, inplace=True)  # Reset the index after dropping rows
display(data.head())
display(data.shape)
```


---

## State Representation

The agent needs a way to "see" the market. We define the **state** as a vector of four features for each day:

- `Close` price
- `SMA_5`
- `SMA_20`
- `Returns`


### State Extraction Function

```python
def get_state(data, index):
    return np.array([
        data.iloc[index, data.columns.get_loc('Close')],
        data.iloc[index, data.columns.get_loc('SMA_5')],
        data.iloc[index, data.columns.get_loc('SMA_20')],
        data.iloc[index, data.columns.get_loc('Returns')]
    ], dtype=np.float32)
```

- **How it works:**
    - For a given day (`index`), this function returns a NumPy array of the four features.

---

## Trading Environment

We need an environment for the agent to interact with, similar to how OpenAI Gym environments work.

### Environment Class

```python
class trading_environment:
    def __init__(self, data):
        self.data = data
        self.initial_balance = 10000
        self.balance = self.initial_balance
        self.holdings = 0      # Number of shares held
        self.index = 0         # Current day in data

    def reset(self):
        self.balance = self.initial_balance
        self.holdings = 0
        self.index = 0
        return get_state(self.data, self.index)

    def step(self, action):
        price = float(self.data.iloc[self.index]['Close'])
        reward = 0

        # Action meanings: 0 = Hold, 1 = Buy, 2 = Sell
        if action == 1 and self.balance &gt;= price:
            self.holdings = self.balance // price
            self.balance -= self.holdings * price
        elif action == 2 and self.holdings &gt; 0:
            self.balance += self.holdings * price
            self.holdings = 0

        self.index += 1
        done = self.index &gt;= len(self.data)
        if done:
            reward = self.balance - self.initial_balance
            next_state = np.zeros(4)  # End of episode
        else:
            next_state = get_state(self.data, self.index)

        return next_state, reward, done, {}
```

- **Key Points:**
    - The agent starts with a fixed cash balance.
    - It can buy, sell, or hold shares.
    - The episode ends when the data runs out.
    - The reward is the profit (or loss) at the end.

---

## Deep Q-Network (DQN) Model

We use a neural network to approximate the Q-value function, which tells the agent the value of each action in each state.

### DQN Architecture

```python
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Ensure input shape is (batch_size, state_size)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
```

- **How it works:**
    - Input: state vector (length 4)
    - Output: Q-values for each action (Hold, Buy, Sell)

---

## DQN Agent

The agent interacts with the environment, stores experiences, and learns from them.

### DQNAgent Class

```python
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)  # Experience replay buffer
        self.gamma = 0.95                 # Discount factor
        self.epsilon = 1.0                # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        self.model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if random.uniform(0, 1) &lt;= self.epsilon:
            return random.randrange(self.action_size)  # Explore
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()           # Exploit

    def replay(self, batch_size):
        if len(self.memory) &lt; batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
                target = reward + self.gamma * torch.max(self.model(next_state)).item()
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            target_tensor = self.model(state_tensor).clone().detach()
            target_tensor[^0][action] = target

            self.optimizer.zero_grad()
            output = self.model(state_tensor)
            loss = self.criterion(output, target_tensor)
            loss.backward()
            self.optimizer.step()

        if self.epsilon &gt; self.epsilon_min:
            self.epsilon *= self.epsilon_decay
```

- **Key Concepts:**
    - **Experience Replay:** Stores past experiences to break correlations and stabilize training.
    - **Epsilon-Greedy Policy:** Balances exploration (random actions) and exploitation (best-known actions).
    - **Q-Learning Update:** Updates the Q-value for the chosen action based on the reward and the estimated value of the next state.

---

## Training the Agent

```python
env = trading_environment(data)
agent = DQNAgent(state_size=4, action_size=3)
batch_size = 32
episodes = 100
total_rewards = []

for episode in range(episodes):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

    agent.replay(batch_size)
    total_rewards.append(total_reward)
    print(f"Episode: {episode+1}/{episodes}, Total Reward: {total_reward}")

print("Training Complete")
```

- **What’s happening?**
    - The agent interacts with the environment, collects experiences, and learns from them after each episode.

---

## Testing the Trained Agent

After training, we simulate a trading session using the trained agent, always choosing the best action (no exploration):

```python
test_env = trading_environment(data)
state = test_env.reset()
done = False

while not done:
    action = agent.act(state)  # Exploit learned policy
    next_state, reward, done, _ = test_env.step(action)
    state = next_state if next_state is not None else state
    print(f"Action: {ACTIONS[action]}, Reward: {reward}")

final_balance = test_env.balance
profit = final_balance - test_env.initial_balance
print(f"Final Balance after testing: ${final_balance:.2f}")
print(f"Total profit: ${profit:.2f}")
```

- **Goal:**
    - See how much profit (or loss) the agent makes after learning.

---

## Summary

- **We built a deep reinforcement learning agent** that learns to trade stocks using historical data.
- **Key steps included:**
    - Downloading and preparing data
    - Engineering features
    - Defining a state representation
    - Creating a trading environment
    - Implementing a DQN model and agent
    - Training and evaluating the agent
- **Concepts covered:**
    - Experience replay, epsilon-greedy exploration, Q-learning updates, and neural network function approximation.

---

## Further Reading

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [OpenAI Gym Reinforcement Learning Environments](https://www.gymlibrary.dev/)
- [Deep Q-Learning Paper](https://www.nature.com/articles/nature14236)
- [yfinance Documentation](https://pypi.org/project/yfinance/)

---

**Congratulations!**
You now have a working DQN-based stock trading agent and a solid understanding of the logic behind each component. Experiment with different stocks, features, and neural network architectures to improve performance!
