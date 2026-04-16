# ==============================
# MARKOV DECISION PROCESS (MDP)
# ==============================

import numpy as np

# States
states = ["S1", "S2", "S3"]

# Actions
actions = ["A1", "A2"]

# Transition Probability Matrix
# P[state][action][next_state]
P = {
    "S1": {
        "A1": [0.7, 0.3, 0.0],
        "A2": [0.4, 0.6, 0.0]
    },
    "S2": {
        "A1": [0.0, 0.8, 0.2],
        "A2": [0.0, 0.5, 0.5]
    },
    "S3": {
        "A1": [0.0, 0.0, 1.0],
        "A2": [0.0, 0.0, 1.0]
    }
}

# Rewards
R = {
    "S1": 5,
    "S2": 10,
    "S3": 0
}

# Discount factor
gamma = 0.9

# Initialize Value Function
V = {s: 0 for s in states}

# Value Iteration
for i in range(10):
    new_V = {}
    for s in states:
        action_values = []
        for a in actions:
            value = 0
            for i_s, s_next in enumerate(states):
                value += P[s][a][i_s] * (R[s_next] + gamma * V[s_next])
            action_values.append(value)
        new_V[s] = max(action_values)
    V = new_V

print("\n--- MDP Value Function ---")
for state in V:
    print(state, ":", round(V[state], 2))