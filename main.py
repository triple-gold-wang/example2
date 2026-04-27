import torch
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader

# 导入你之前写好的各个模块
from geometry import DomainSampler
from model import SolidDeepONet
from loss import compute_dem_loss

# ==========================================
# ⚙️ 1. 全局超参数与环境配置
# ==========================================
# 设备配置：如果你的电脑有NVIDIA显卡且装了CUDA，会自动用GPU，否则用CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"当前使用的计算设备: {device}")

# 物理与几何配置
L_CONFIG = 1.0       # 正方形半边长
R0_CONFIG = 0.3      # 参考域圆孔半径
E_CONFIG = 1.0       # 无量纲杨氏模量
NU_CONFIG = 0.3      # 泊松比
TX_CONFIG = 10.0     # 右边界拉力

# 训练配置
EPOCHS = 100         # 总训练轮次
BATCH_SIZE = 32      # 批次大小
LEARNING_RATE = 1e-3 # 学习率 (Adam优化器默认通常是 1e-3)

# 采样配置
N_INTERIOR = 3000    # 内部参考域撒点数 (为了测试速度，先用 3000)
N_BND_RIGHT = 200    # 右边界撒点数 (算外力做功)

# ==========================================
# 📊 2. 数据准备阶段
# ==========================================
print("\n--- 正在加载数据与采样 ---")

# 2.1 加载几何参数 (a, b, theta)
try:
    df_params = pd.read_csv('geometry_params_200.csv')
    # 转换为 PyTorch 张量
    params_tensor = torch.tensor(df_params.values, dtype=torch.float32)
except FileNotFoundError:
    print("错误: 找不到 geometry_params_200.csv！请先运行 geo_data.py。")
    exit()

# 2.2 构建 DataLoader 实现自动批处理
dataset = TensorDataset(params_tensor)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
print(f"成功加载几何参数，共 {len(dataset)} 个样本。")

# 2.3 生成参考域坐标点 (这部分是固定的，所以只在循环外生成一次！)
sampler = DomainSampler(L=L_CONFIG, R0=R0_CONFIG)
X_inner = sampler.sample_interior(N_INTERIOR).to(device)
X_right = sampler.sample_right_boundary(N_BND_RIGHT).to(device)

print(f"参考域采样完成: 内部点 {X_inner.shape[0]} 个, 右边界点 {X_right.shape[0]} 个。")

# ==========================================
# 🧠 3. 模型与优化器初始化
# ==========================================
print("\n--- 正在初始化模型 ---")

# 定义网络结构 (你可以根据需要增减层数)
branch_layers = [3, 64, 128, 100]  # 输入维度 3
trunk_layers = [2, 64, 128, 100]   # 输入维度 2

# 实例化模型并移至对应设备
model = SolidDeepONet(branch_layers, trunk_layers, L=L_CONFIG).to(device)

# 实例化优化器 (Adam 是最常用的起手式)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ==========================================
# 🏃‍♂️ 4. 主训练循环 (Training Loop)
# ==========================================
print("\n--- 开始训练 ---")

for epoch in range(EPOCHS):
    epoch_loss = 0.0
    
    # 遍历数据加载器中的每一个小批次
    for batch_idx, batch_data in enumerate(dataloader):
        
        # 提取当前批次的几何参数 [batch_size, 3] 并发送到设备
        params_batch = batch_data[0].to(device)
        
        # 1. 梯度清零 (PyTorch 必须的常规操作)
        optimizer.zero_grad()
        
        # 2. 计算 DEM 损失 (调用 loss.py)
        loss = compute_dem_loss(
            model=model, 
            params=params_batch, 
            X_inner=X_inner, 
            X_right=X_right,
            L=L_CONFIG, R0=R0_CONFIG, 
            E=E_CONFIG, nu=NU_CONFIG, Tx=TX_CONFIG
        )
        
        # 3. 反向传播 (自动微分计算梯度)
        loss.backward()
        
        # 4. 更新权重
        optimizer.step()
        
        # 累加损失用于监控
        epoch_loss += loss.item()
    
    # 每 10 个 Epoch 打印一次信息
    if (epoch + 1) % 10 == 0 or epoch == 0:
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{EPOCHS}] | 平均 Total Potential Energy Loss: {avg_loss:.6f}")

print("\n🎉 测试训练流程运行完毕！网络能够正常更新！")
# 可选：保存模型权重
torch.save(model.state_dict(), 'deeponet_dem_test.pth')