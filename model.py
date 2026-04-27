import torch
import torch.nn as nn

class FNN(nn.Module):
    """基础全连接前馈神经网络 (Trunk 和 Branch 的基础组件)"""
    def __init__(self, layers):
        super().__init__()
        self.net = nn.Sequential()
        for i in range(len(layers)-2):
            self.net.add_module(f'linear_{i}', nn.Linear(layers[i], layers[i+1]))
            # 在 SciML 中，Tanh 或 GELU 比 ReLU 的二阶导数表现更好
            self.net.add_module(f'act_{i}', nn.Tanh()) 
        self.net.add_module(f'linear_out', nn.Linear(layers[-2], layers[-1]))

    def forward(self, x):
        return self.net(x)

class SolidDeepONet(nn.Module):
    def __init__(self, branch_layers, trunk_layers, L=1.0):
        """
        branch_layers: 例如 [3, 64, 128, p*2] (输入 a, b, theta)
        trunk_layers: 例如 [2, 64, 128, p*2] (输入 X, Y)
        L: 正方形半边长，用于边界约束
        """
        super().__init__()
        self.branch = FNN(branch_layers)
        self.trunk = FNN(trunk_layers)
        self.L = L

    def forward(self, params, X_ref):
        """
        params: [B, 3] -> 几何参数 (a, b, theta)
        X_ref: 可能是 [N, 2] (单个坐标网格)，也可能是 [B, N, 2] (批量坐标网格)
        """
        # 1. 提取特征
        B_out = self.branch(params) # [B, p*2]
        T_out = self.trunk(X_ref)   # [N, p*2] 或 [B, N, p*2]
        
        # 2. 将特征对半分给 u 和 v
        p = B_out.shape[1] // 2
        B_u, B_v = B_out[:, :p], B_out[:, p:] # [B, p]
        
        # 使用 ... (Ellipsis) 可以自适应切片，无论前面有多少个维度
        T_u, T_v = T_out[..., :p], T_out[..., p:] # [N, p] 或 [B, N, p]
        
        # 3. 算子内积融合 & 4. 硬约束 Ansatz
        # 判断 Trunk 输出的维度，选择对应的操作
        if T_out.dim() == 2:
            # 针对 2D 输入 [N, p] (例如早期的 Dummy 测试)
            u_raw = torch.einsum('bp,np->bn', B_u, T_u)
            v_raw = torch.einsum('bp,np->bn', B_v, T_v)
            
            X_coord = X_ref[:, 0] # [N]
            distance_func = (X_coord + self.L).unsqueeze(0) # [1, N]
            
        else:
            # 针对 3D 输入 [B, N, p] (真实训练的 batched 模式)
            u_raw = torch.einsum('bp,bnp->bn', B_u, T_u)
            v_raw = torch.einsum('bp,bnp->bn', B_v, T_v)
            
            X_coord = X_ref[:, :, 0] # [B, N]
            distance_func = X_coord + self.L # [B, N]
        
        # 应用距离函数强行将左侧边界位移归零
        u_final = distance_func * u_raw
        v_final = distance_func * v_raw
        
        # 5. 组合输出位移 [B, N, 2]
        return torch.stack([u_final, v_final], dim=-1)
    

if __name__ == '__main__':
    branch_layers = [3, 64, 128, 100] 
    # 设定 Trunk 网络层数: 输入2个坐标 (X,Y), 隐藏层 64, 128, 最后的特征维度也是 100
    trunk_layers = [2, 64, 128, 100] 

    # 实例化模型，传入你的正方形半边长 L (用于硬约束)
    L_val = 1.0
    model = SolidDeepONet(branch_layers, trunk_layers, L=L_val)
    print("模型实例化成功！\n")
    batch_size = 32   # 假设我们一次性输入 32 种不同的椭圆形状
    N_points = 5000   # 假设我们在参考域撒了 5000 个点

    # Branch 的 Dummy 输入：维度 [batch_size, 3]
    # 使用 torch.rand 生成 [0, 1) 之间的均匀分布随机数模拟参数
    dummy_params = torch.rand((batch_size, 3)) 

    # Trunk 的 Dummy 输入：维度 [N_points, 2]
    # 使用 torch.randn 生成标准正态分布随机数模拟坐标
    dummy_X_ref = torch.randn((N_points, 2)) 

    print(f"Dummy Params 维度: {dummy_params.shape}")
    print(f"Dummy X_ref 维度: {dummy_X_ref.shape}\n")

    # --- 3. 将 Dummy 张量输入网络进行测试 ---
    # 前向传播
    output_displacement = model(dummy_params, dummy_X_ref)

    print(f"网络输出位移的维度: {output_displacement.shape}")
    # 期望输出应该是 [32, 5000, 2]，代表 32个形状下，5000个点的 (u, v) 位移