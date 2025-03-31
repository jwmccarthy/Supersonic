import torch
import supersonic as ssl

print("Creating envs...")

num_envs = 1024
envs = ssl.EnvironmentManager(num_envs)

print("Envs created.")

actions = torch.rand(num_envs, device="cuda", dtype=torch.float32)

print(actions)

states = envs.reset()

print(states)

for _ in range(1000):
    states = envs.step(actions)

print(states)
print(states.shape)
print(states.device)