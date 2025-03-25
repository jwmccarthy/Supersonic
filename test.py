import torch
import supersonic as ssl

num_envs = 1024
envs = ssl.EnvironmentManager(num_envs)

actions = torch.rand(num_envs, device="cuda", dtype=torch.float32)

# Step environments in parallel
envs.step(actions)

# Retrieve states (on GPU)
states = envs.get_states()
print(states.shape)
print(states.device)  # Should print: cuda:0
