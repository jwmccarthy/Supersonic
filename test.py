import torch
import supersonic as ssl

num_envs = 1024
envs = ssl.EnvironmentManager(num_envs)

actions = torch.rand(num_envs, device="cuda", dtype=torch.float32)

states = envs.step(actions)

print(states)

states = envs.reset()

print(states)

print(states.shape)
print(states.device)  # Should print: cuda:0