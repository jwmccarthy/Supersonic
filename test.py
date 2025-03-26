import torch
import supersonic as ssl

num_envs = 1024
envs = ssl.EnvironmentManager(num_envs)

actions = torch.rand(num_envs, device="cuda", dtype=torch.float32)

print(actions)

states = envs.reset()

print(states)

for i in range(1000):
    states = envs.step(actions)

print(states)

print(states.shape)
print(states.device)