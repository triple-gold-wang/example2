import numpy as np
import torch
import matplotlib.pyplot as plt

L_CONFIG = 1.0       # 正方形半边长
R0_CONFIG = 0.3      # 参考域圆孔半径

# 2. 采样数量参数 (这里你可以直接改成 5000)
N_INTERIOR = 5000    # 内部参考域撒点数
N_BND_LEFT = 200     # 左边界约束撒点数
N_BND_RIGHT = 200    # 右边界载荷撒点数

# 3. 目标形状参数测试用例
TEST_A = 0.7         # 目标椭圆长半轴
TEST_B = 0.3         # 目标椭圆短半轴
TEST_THETA = np.pi/2 # 目标椭圆旋转角


class GeometricMapper:
    def __init__(self, L, R0):
        """
        L: 正方形半边长
        R0: 参考域圆孔半径
        """
        self.L = L
        self.R0 = R0

    def get_square_radius(self, theta):
        """计算正方形边界在theta方向的极径"""
        theta = theta % (2 * np.pi)
        # 将角度映射到 [-pi/4, pi/4] 及其对称区间
        a = np.abs(np.cos(theta))
        b = np.abs(np.sin(theta))
        return self.L / np.maximum(a, b)

    def get_ellipse_radius(self, theta, a, b, alpha):
        """计算旋转alpha角度后的椭圆在theta方向的极径"""
        # 旋转坐标系
        phi = theta - alpha
        r = (a * b) / np.sqrt((b * np.cos(phi))**2 + (a * np.sin(phi))**2)
        return r

    def map_points(self, X_ref, a, b, alpha):
        """
        将参考域点映射到目标域
        X_ref: [N, 2] 的张量 (X, Y)
        """
        X = X_ref[:, 0]
        Y = X_ref[:, 1]
        
        R = torch.sqrt(X**2 + Y**2)
        Theta = torch.atan2(Y, X)
        
        R_poly = self.get_square_radius(Theta)
        R_ell = self.get_ellipse_radius(Theta, a, b, alpha)
        
        # 线性插值映射:保持外边界R_poly不变，将R0映射为R_ell
        # r = R_ell + (R - R0) * (R_poly - R_ell) / (R_poly - R0)
        # 注意：这里假设所有采样点都在 R0 和 R_poly 之间
        ratio = (R - self.R0) / (R_poly - self.R0)
        r_target = R_ell + ratio * (R_poly - R_ell)
        
        x_target = r_target * torch.cos(Theta)
        y_target = r_target * torch.sin(Theta)
        
        return torch.stack([x_target, y_target], dim=1)

class DomainSampler:
    def __init__(self, L, R0):
        self.L = L
        self.R0 = R0
        
        # 初始化 Sobol 引擎
        # scramble=True 表示使用 Owen 扰动，这样每次实例化或 draw 出来的序列既保持均匀性，又带有随机性（防止网络过拟合到固定的点）
        self.sobol_2d = torch.quasirandom.SobolEngine(dimension=2, scramble=True)
        self.sobol_1d_left = torch.quasirandom.SobolEngine(dimension=1, scramble=True)
        self.sobol_1d_right = torch.quasirandom.SobolEngine(dimension=1, scramble=True)

    def sample_interior(self, n_points):
        """在参考域（圆孔正方形）内使用 Sobol 序列均匀采样"""
        valid_points = []
        collected = 0
        
        # 由于要剔除圆孔内的点，需要生成多于 n_points 的样本以避免多次循环
        # 正方形面积 4*L^2，圆孔面积 pi*R0^2。这里做一个简单的放大系数估计
        area_ratio = (4 * self.L**2) / (4 * self.L**2 - torch.pi * self.R0**2)
        batch_size = int(n_points * area_ratio * 1.1) # 增加 10% 的余量保证一次性采够

        while collected < n_points:
            # Sobol 默认生成 [0, 1] 范围的值，将其映射到 [-L, L]
            pts = self.sobol_2d.draw(batch_size) * 2 * self.L - self.L
            
            # 计算每个点到原点的距离
            r = torch.linalg.norm(pts, dim=1)
            
            # 过滤出圆孔外部的点
            mask = r >= self.R0
            pts_valid = pts[mask]
            
            valid_points.append(pts_valid)
            collected += len(pts_valid)
        
        # 将列表中的 tensor 拼接，并严格截取你需要的前 n_points 个点
        return torch.cat(valid_points, dim=0)[:n_points]

    def sample_left_boundary(self, n_points):
        """使用 Sobol 序列采样左边界 (X = -L) 用于位移约束"""
        # 一维 Sobol 序列生成 [0, 1] 的点，映射到 Y 轴的 [-L, L]
        y = self.sobol_1d_left.draw(n_points) * 2 * self.L - self.L
        # X 坐标全为 -L
        x = torch.full_like(y, -self.L)
        return torch.cat([x, y], dim=1)

    def sample_right_boundary(self, n_points):
        """使用 Sobol 序列采样右边界 (X = L) 用于力载荷"""
        y = self.sobol_1d_right.draw(n_points) * 2 * self.L - self.L
        x = torch.full_like(y, self.L)
        return torch.cat([x, y], dim=1)

if __name__ == '__main__':
    sampler = DomainSampler(L=L_CONFIG, R0=R0_CONFIG)
    mapper = GeometricMapper(L=L_CONFIG, R0=R0_CONFIG)

# 1. 采集参考域点 (传入配置的点数变量)
    X_inner = sampler.sample_interior(N_INTERIOR)
    X_left = sampler.sample_left_boundary(N_BND_LEFT)
    X_right = sampler.sample_right_boundary(N_BND_RIGHT)

    print(f"成功采样内部点数量: {X_inner.shape[0]}")

# 2. 映射到目标域 (传入配置的形状变量)
    x_inner_mapped = mapper.map_points(X_inner, TEST_A, TEST_B, TEST_THETA)
    x_left_mapped = mapper.map_points(X_left, TEST_A, TEST_B, TEST_THETA)
    x_right_mapped = mapper.map_points(X_right, TEST_A, TEST_B, TEST_THETA)

    # 4. 可视化
    plt.figure(figsize=(8,8))
    plt.scatter(x_inner_mapped[:,0], x_inner_mapped[:,1], s=1, label='Interior') # 点变多了，把 s (size) 调小一点
    plt.scatter(x_left_mapped[:,0], x_left_mapped[:,1], s=10, c='r', label='Fixed Boundary (Left)')
    plt.scatter(x_right_mapped[:,0], x_right_mapped[:,1], s=10, c='g', label='Traction Boundary (Right)')
    plt.axis('equal')
    plt.title(f"Mapped Domain (N={N_INTERIOR}, a={TEST_A}, b={TEST_B}, theta={TEST_THETA:.2f})")
    plt.legend()
    plt.show()