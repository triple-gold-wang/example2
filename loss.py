import torch

# ==========================================
# 1. 辅助求导函数
# ==========================================
def get_gradient(y, x):
    """
    计算张量 y 对张量 x 的梯度
    y: [batch_size, N]
    x: [batch_size, N, 2]
    返回: [batch_size, N, 2] 包含 dy/dx_1, dy/dx_2
    """
    # y.sum() 是一个标量，但 autograd 会针对每个 x 元素分别求导，这是求 batch 梯度的标准做法
    grad = torch.autograd.grad(y.sum(), x, create_graph=True)[0]
    return grad

# ==========================================
# 2. 纯 PyTorch 批量几何映射 (Autograd 必须)
# ==========================================
def pure_torch_batched_mapping(X_batch, params, L=1.0, R0=0.3):
    """
    全量支持 Batch 和 Autograd 的映射函数。
    X_batch: [batch_size, N, 2]
    params: [batch_size, 3] 分别是 a, b, alpha
    """
    X = X_batch[..., 0] # [batch_size, N]
    Y = X_batch[..., 1]
    
    # 将参数扩展维度以匹配坐标张量
    a = params[:, 0].unsqueeze(1)     # [batch_size, 1]
    b = params[:, 1].unsqueeze(1)
    alpha = params[:, 2].unsqueeze(1)

    R = torch.sqrt(X**2 + Y**2)
    Theta = torch.atan2(Y, X)

    # --- 计算正方形外径 R_poly ---
    Theta_mod = Theta % (2 * torch.pi)
    cos_val = torch.abs(torch.cos(Theta_mod))
    sin_val = torch.abs(torch.sin(Theta_mod))
    R_poly = L / torch.maximum(cos_val, sin_val)

    # --- 计算椭圆内径 R_ell ---
    phi = Theta - alpha
    R_ell = (a * b) / torch.sqrt((b * torch.cos(phi))**2 + (a * torch.sin(phi))**2)

    # --- 径向插值 ---
    ratio = (R - R0) / (R_poly - R0)
    r_target = R_ell + ratio * (R_poly - R_ell)

    x_target = r_target * torch.cos(Theta)
    y_target = r_target * torch.sin(Theta)

    return torch.stack([x_target, y_target], dim=-1) # [batch_size, N, 2]

# ==========================================
# 3. 核心 DEM Loss 计算
# ==========================================
def compute_dem_loss(model, params, X_inner, X_right, L=1.0, R0=0.3, E=1.0, nu=0.3, Tx=10.0):
    """
    params: [B, 3] 的几何参数 batch
    X_inner: [N_in, 2] 的内部采样点
    X_right: [N_right, 2] 的右边界采样点
    Tx: 右侧施加的均布拉力载荷大小
    """
    batch_size = params.shape[0]
    
    # 1. 扩展坐标以匹配 Batch Size，并必须开启 require_grad
    # X_batch 形状变为 [B, N_in, 2]
    X_batch = X_inner.unsqueeze(0).expand(batch_size, -1, -1).clone().requires_grad_(True)
    
    # 右边界点只用来算做功，不需要求坐标梯度，不需要 require_grad
    X_right_batch = X_right.unsqueeze(0).expand(batch_size, -1, -1)

    # ===============================
    # Step A: 前向传播
    # ===============================
    # 将内部点和边界点拼接一次性送入网络，提高速度
    N_in = X_batch.shape[1]
    X_eval = torch.cat([X_batch, X_right_batch], dim=1) # [B, N_in + N_right, 2]
    
    U_pred = model(params, X_eval) # [B, N_in + N_right, 2]
    
    # 拆分预测结果
    U_inner = U_pred[:, :N_in, :]    # [B, N_in, 2]
    U_right = U_pred[:, N_in:, :]    # [B, N_right, 2]

    u = U_inner[..., 0] # [B, N_in]
    v = U_inner[..., 1]

    # ===============================
    # Step B: 计算基础偏导数 (Autograd)
    # ===============================
    # 1. 位移对参考域坐标的导数
    du_dX_Y = get_gradient(u, X_batch) # 返回 [B, N_in, 2]
    du_dX, du_dY = du_dX_Y[..., 0], du_dX_Y[..., 1]
    
    dv_dX_Y = get_gradient(v, X_batch)
    dv_dX, dv_dY = dv_dX_Y[..., 0], dv_dX_Y[..., 1]

    # 2. 生成目标域坐标，并求雅可比矩阵
    x_target = pure_torch_batched_mapping(X_batch, params, L, R0) # [B, N_in, 2]
    x = x_target[..., 0]
    y = x_target[..., 1]

    dx_dX_Y = get_gradient(x, X_batch)
    J11, J12 = dx_dX_Y[..., 0], dx_dX_Y[..., 1]

    dy_dX_Y = get_gradient(y, X_batch)
    J21, J22 = dy_dX_Y[..., 0], dy_dX_Y[..., 1]

    detJ = J11*J22 - J12*J21 # 雅可比行列式 [B, N_in]

    # ===============================
    # Step C: 计算真实应变 (链式法则)
    # ===============================
    du_dx = (du_dX * J22 - du_dY * J21) / detJ
    du_dy = (-du_dX * J12 + du_dY * J11) / detJ
    dv_dx = (dv_dX * J22 - dv_dY * J21) / detJ
    dv_dy = (-dv_dX * J12 + dv_dY * J11) / detJ

    eps_xx = du_dx
    eps_yy = dv_dy
    eps_xy = 0.5 * (du_dy + dv_dx)

    # ===============================
    # Step D: 计算应变能 (平面应力)
    # ===============================
    C = E / (2 * (1 - nu**2))
    W = C * (eps_xx**2 + eps_yy**2 + 2*nu*eps_xx*eps_yy + (1 - nu)*eps_xy**2) # [B, N_in]
    
    # 蒙特卡洛积分：能量密度 * 雅可比行列式，然后在参考域求均值再乘面积
    Area_ref = 4 * (L**2) - torch.pi * (R0**2)
    # mean(dim=1) 是对 N_in 个点求平均，输出形状为 [B]
    Strain_Energy = (W * detJ).mean(dim=1) * Area_ref 

    # ===============================
    # Step E: 计算外力做功
    # ===============================
    # 提取右边界 X 轴方向的位移 u_right
    u_right = U_right[..., 0] # [B, N_right]
    Length_right = 2 * L
    Work_External = (Tx * u_right).mean(dim=1) * Length_right # [B]

    # ===============================
    # Step F: 总势能 Loss
    # ===============================
    Total_Potential_Energy = Strain_Energy - Work_External
    
    # 损失函数是总势能的期望值（取 batch 的平均）
    loss = Total_Potential_Energy.mean()

    return loss