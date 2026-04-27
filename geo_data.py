import numpy as np
import pandas as pd
from scipy.stats import qmc

# 使用拉丁超立方抽样生成 200 个样本
num_samples = 200
sampler = qmc.LatinHypercube(d=3)
sample = sampler.random(n=num_samples)

# 映射到具体的物理区间
# a: [0.2, 0.8], b: [0.1, 0.8], theta: [0, pi]
a_bounds = [0.2, 0.8]
b_bounds = [0.1, 0.8]
theta_bounds = [0, np.pi]

a = a_bounds[0] + sample[:, 0] * (a_bounds[1] - a_bounds[0])
b = b_bounds[0] + sample[:, 1] * (b_bounds[1] - b_bounds[0])
# 强制让 a >= b，如果不满足则交换
for i in range(num_samples):
    if b[i] > a[i]:
        a[i], b[i] = b[i], a[i]
        
theta = theta_bounds[0] + sample[:, 2] * (theta_bounds[1] - theta_bounds[0])

# 保存为 DataFrame 并导出 CSV
df_params = pd.DataFrame({'a': a, 'b': b, 'theta': theta})
df_params.to_csv('geometry_params_200.csv', index=False)
