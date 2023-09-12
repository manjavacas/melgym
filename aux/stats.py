#!/usr/bin/env python

import os
import pandas as pd

metrics_dir = '../ep_metrics/'

reward_sums = {}

for file_csv in os.listdir(metrics_dir):
    if file_csv.endswith('.csv'):
        file_path = os.path.join(metrics_dir, file_csv)

        df = pd.read_csv(file_path)
        reward_sum = df['reward'].sum()
        reward_sums[file_csv] = reward_sum

max_reward_file = max(reward_sums, key=reward_sums.get)  # type: ignore

max_reward_file_path = os.path.join(metrics_dir, max_reward_file)
df_max_reward = pd.read_csv(max_reward_file_path)

mean_action_max_reward = df_max_reward['action'].mean()

print(f'Max reward .csv: {max_reward_file}')
print(f'Reward sum: {reward_sums[max_reward_file]:.4f}')
print(f'Mean action: {mean_action_max_reward:.4f}')
