import pandas as pd
import numpy as np
import torch
import os
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.tri import Triangulation
from scipy.interpolate import griddata

# ==========================================
# 0. 前置准备 (请确保导入你写好的类)
# ==========================================
# 从你的代码中导入网络模型和几何映射工具
from model import SolidDeepONet
from geometry import GeometricMapper, DomainSampler

# ==========================================
# 可视化配置（不影响原有评估逻辑）
# ==========================================
ENABLE_VIS = True
VIS_RANDOM_COUNT = 2
VIS_RANDOM_SEED = 2026
VIS_OUTPUT_DIR = 'results_vis'
ABAQUS_CMAP = 'jet'   # Abaqus风格常见彩虹色图
ERROR_CMAP = 'inferno'


def _ellipse_level_set(x, y, a, b, theta):
    """椭圆隐式函数值: <=1 在椭圆内，=1 在边界上。"""
    c, s = np.cos(theta), np.sin(theta)
    xr = c * x + s * y
    yr = -s * x + c * y
    return (xr / a) ** 2 + (yr / b) ** 2


def _build_masked_triangulation(x, y, a, b, theta):
    """构建带孔洞掩膜的三角剖分，避免等值图跨孔填充。"""
    tri = Triangulation(x, y)
    triangles = tri.triangles

    x0 = x[triangles[:, 0]]
    y0 = y[triangles[:, 0]]
    x1 = x[triangles[:, 1]]
    y1 = y[triangles[:, 1]]
    x2 = x[triangles[:, 2]]
    y2 = y[triangles[:, 2]]

    xc = (x0 + x1 + x2) / 3.0
    yc = (y0 + y1 + y2) / 3.0
    xm01 = (x0 + x1) / 2.0
    ym01 = (y0 + y1) / 2.0
    xm12 = (x1 + x2) / 2.0
    ym12 = (y1 + y2) / 2.0
    xm20 = (x2 + x0) / 2.0
    ym20 = (y2 + y0) / 2.0

    in_hole = (
        (_ellipse_level_set(xc, yc, a, b, theta) < 1.0)
        | (_ellipse_level_set(xm01, ym01, a, b, theta) < 1.0)
        | (_ellipse_level_set(xm12, ym12, a, b, theta) < 1.0)
        | (_ellipse_level_set(xm20, ym20, a, b, theta) < 1.0)
    )
    tri.set_mask(in_hole)
    return tri


def _overlay_geometry(ax, a, b, theta, L=1.0):
    """叠加外方形边界和椭圆孔边界，提升几何可读性。"""
    square_x = np.array([-L, L, L, -L, -L])
    square_y = np.array([-L, -L, L, L, -L])
    ax.plot(square_x, square_y, color='black', linewidth=0.9, zorder=8)

    hole = Ellipse(
        xy=(0.0, 0.0),
        width=2.0 * a,
        height=2.0 * b,
        angle=np.degrees(theta),
        facecolor='white',
        edgecolor='black',
        linewidth=1.1,
        zorder=9,
    )
    ax.add_patch(hole)


def _draw_abaqus_contour(ax, tri, values, title, cmap, a, b, theta, vmin=None, vmax=None):
    """使用三角剖分等值云图，接近 Abaqus 结果显示风格。"""
    mappable = ax.tricontourf(tri, values, levels=36, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.tricontour(tri, values, levels=12, colors='k', linewidths=0.15, alpha=0.22)
    _overlay_geometry(ax, a, b, theta, L=1.0)

    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=10)
    return mappable


def save_uv_comparison_plot(sample, output_dir):
    """保存单样本 3x2 图：u/v 的预测-真值对比 + 绝对误差。"""
    x = sample['x']
    y = sample['y']
    a = sample['a']
    b = sample['b']
    theta = sample['theta']

    tri = _build_masked_triangulation(x, y, a, b, theta)

    u_pred = sample['u_pred']
    v_pred = sample['v_pred']
    u_true = sample['u_true']
    v_true = sample['v_true']

    u_err = np.abs(u_pred - u_true)
    v_err = np.abs(v_pred - v_true)

    u_min, u_max = min(u_pred.min(), u_true.min()), max(u_pred.max(), u_true.max())
    v_min, v_max = min(v_pred.min(), v_true.min()), max(v_pred.max(), v_true.max())

    fig, axes = plt.subplots(3, 2, figsize=(11, 14), constrained_layout=True)

    m_u_pred = _draw_abaqus_contour(axes[0, 0], tri, u_pred, 'U Pred', ABAQUS_CMAP, a, b, theta, u_min, u_max)
    _draw_abaqus_contour(axes[0, 1], tri, u_true, 'U True', ABAQUS_CMAP, a, b, theta, u_min, u_max)
    fig.colorbar(m_u_pred, ax=[axes[0, 0], axes[0, 1]], fraction=0.03, pad=0.01)

    m_v_pred = _draw_abaqus_contour(axes[1, 0], tri, v_pred, 'V Pred', ABAQUS_CMAP, a, b, theta, v_min, v_max)
    _draw_abaqus_contour(axes[1, 1], tri, v_true, 'V True', ABAQUS_CMAP, a, b, theta, v_min, v_max)
    fig.colorbar(m_v_pred, ax=[axes[1, 0], axes[1, 1]], fraction=0.03, pad=0.01)

    m_u_err = _draw_abaqus_contour(axes[2, 0], tri, u_err, '|U Error|', ERROR_CMAP, a, b, theta)
    m_v_err = _draw_abaqus_contour(axes[2, 1], tri, v_err, '|V Error|', ERROR_CMAP, a, b, theta)
    fig.colorbar(m_u_err, ax=axes[2, 0], fraction=0.046, pad=0.02)
    fig.colorbar(m_v_err, ax=axes[2, 1], fraction=0.046, pad=0.02)

    title = (
        f"Shape {sample['shape_id']} | a={sample['a']:.3f}, b={sample['b']:.3f}, "
        f"theta={sample['theta']:.3f} | L2={sample['l2_error']*100:.2f}%"
    )
    fig.suptitle(title, fontsize=12)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"shape_{sample['shape_id']:03d}_uv_compare.png")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path

def main():
    print("正在加载模型和采样点...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"评估设备: {device}")
    
    # 1. 实例化网络并加载权重 (替换为你真实的模型加载逻辑)
    branch_layers = [3, 64, 128, 100]  # 输入维度 3
    trunk_layers = [2, 64, 128, 100]   # 输入维度 2
    model = SolidDeepONet(branch_layers, trunk_layers).to(device)
    model.load_state_dict(torch.load('deeponet_dem_test.pth', map_location=device))
    model.eval() # 开启评估模式
    
    # 2. 准备参考域内部点
    sampler = DomainSampler(L=1.0, R0=0.3)
    mapper = GeometricMapper(L=1.0, R0=0.3)
    X_inner = sampler.sample_interior(3000)
    X_inner_model = X_inner.to(device)
    
    # ==========================================
    # 1. 读取主索引并截取前 79 个形状
    # ==========================================
    csv_index_path = 'geometry_params_200.csv'
    abaqus_data_dir = os.path.join('gt')
    
    if not os.path.exists(csv_index_path):
        print(f"找不到索引文件: {csv_index_path}")
        return
        
    master_params = pd.read_csv(csv_index_path)
    # 仅选取前 79 个形状 (索引 0 到 78)
    test_params = master_params.head(79)
    
    total_l2_errors = []
    success_count = 0
    vis_samples = []
    
    print("\n=============================================")
    print("开始进行前 79 个形状的自动化误差对比评估...")
    print("=============================================\n")

    # ==========================================
    # 2. 自动化评估循环
    # ==========================================
    for index, row in test_params.iterrows():
        shape_id = int(row['shape_id'])
        a = row['a']
        b = row['b']
        theta = row['theta']
        
        target_csv_path = os.path.join(abaqus_data_dir, f'abaqus_truth_{shape_id}.csv')
        
        if not os.path.exists(target_csv_path):
            print(f"警告：找不到形状 ID {shape_id} 的真值文件，已跳过...")
            continue
            
        # --- A. 神经网络预测 ---
        # 构造分支网络输入参数张量
        params_tensor = torch.tensor([[a, b, theta]], dtype=torch.float32, device=device)
        
        with torch.no_grad():
            u_pred_tensor = model(params_tensor, X_inner_model).squeeze(0)
            x_target_tensor = mapper.map_points(X_inner, a, b, theta)
            
        x_nn = x_target_tensor[:, 0].detach().cpu().numpy()
        y_nn = x_target_tensor[:, 1].detach().cpu().numpy()
        u_pred = u_pred_tensor[:, 0].detach().cpu().numpy()
        v_pred = u_pred_tensor[:, 1].detach().cpu().numpy()
        
        
        # --- B. 读取 Abaqus 真值 ---
        try:
            df_abq = pd.read_csv(target_csv_path)
            x_abq = df_abq['x'].values
            y_abq = df_abq['y'].values
            u_abq = df_abq['U1'].values
            v_abq = df_abq['U2'].values
        except Exception as e:
            print(f"读取 CSV 文件 {target_csv_path} 时出错: {e}")
            continue
            
        # --- C. 空间插值对齐 ---
        # 利用 griddata 将 Abaqus 不规则网格插值到神经网络点云上
        u_true_aligned = griddata((x_abq, y_abq), u_abq, (x_nn, y_nn), method='linear', fill_value=np.nan)
        v_true_aligned = griddata((x_abq, y_abq), v_abq, (x_nn, y_nn), method='linear', fill_value=np.nan)
        
        # --- D. 计算宏观相对 L2 误差 ---
        valid_mask = np.isfinite(u_true_aligned) & np.isfinite(v_true_aligned)
        valid_count = int(valid_mask.sum())
        if valid_count < 100:
            print(f"警告：Shape ID {shape_id} 的有效对齐点过少（{valid_count}），已跳过该样本。")
            continue

        pred_vec = np.column_stack([u_pred[valid_mask], v_pred[valid_mask]])
        true_vec = np.column_stack([u_true_aligned[valid_mask], v_true_aligned[valid_mask]])
        diff_vec = pred_vec - true_vec
        denom = np.linalg.norm(true_vec)
        if denom < 1e-12:
            print(f"警告：Shape ID {shape_id} 的真值位移范数接近 0，已跳过该样本。")
            continue
        l2_error = np.linalg.norm(diff_vec) / denom
        
        print(f"Shape ID: {shape_id:>2} | a={a:.3f}, b={b:.3f}, theta={theta:.3f} | L2 Error: {l2_error*100:>5.2f}%")
        total_l2_errors.append(l2_error)
        success_count += 1

        # 收集可视化数据（仅在有效点上，避免 NaN）
        vis_samples.append({
            'shape_id': shape_id,
            'a': a,
            'b': b,
            'theta': theta,
            'l2_error': l2_error,
            'valid_count': valid_count,
            'x': x_nn[valid_mask],
            'y': y_nn[valid_mask],
            'u_pred': u_pred[valid_mask],
            'v_pred': v_pred[valid_mask],
            'u_true': u_true_aligned[valid_mask],
            'v_true': v_true_aligned[valid_mask],
        })

    # ==========================================
    # 3. 输出模型整体宏观表现
    # ==========================================
    print("\n=============================================")
    print(f"评估完成！共成功处理 {success_count} 个有效样本。")
    if total_l2_errors:
        mean_error = np.mean(total_l2_errors)
        max_error = np.max(total_l2_errors)
        min_error = np.min(total_l2_errors)
        print(f"平均相对 L2 误差 : **{mean_error*100:.2f}%**")
        print(f"最小相对 L2 误差 : {min_error*100:.2f}%")
        print(f"最大相对 L2 误差 : {max_error*100:.2f}%")
        
        if mean_error < 0.05:
            print("结论：模型精度极高（平均误差 < 5%），深度能量法完全收敛！")
        elif mean_error < 0.15:
            print("结论：模型具备良好的物理捕捉能力，可能需要增加训练轮数或微调学习率。")
        else:
            print("结论：误差偏大，请重点排查坐标对齐、载荷量纲一致性或网络容量问题。")
    print("=============================================\n")

    # ==========================================
    # 4. 导出可视化：固定1个最大误差样本 + 随机2个样本
    # ==========================================
    if ENABLE_VIS and vis_samples:
        max_err_sample = max(vis_samples, key=lambda s: s['l2_error'])
        remaining = [s for s in vis_samples if s['shape_id'] != max_err_sample['shape_id']]

        rng = random.Random(VIS_RANDOM_SEED)
        random_count = min(VIS_RANDOM_COUNT, len(remaining))
        random_samples = rng.sample(remaining, random_count) if random_count > 0 else []

        selected = [max_err_sample] + random_samples

        print("开始导出可视化图片（Abaqus风格）...")
        print(f"固定最大误差样本: Shape {max_err_sample['shape_id']} (L2={max_err_sample['l2_error']*100:.2f}%)")
        if random_samples:
            print("随机样本: " + ", ".join(str(s['shape_id']) for s in random_samples))

        for sample in selected:
            img_path = save_uv_comparison_plot(sample, VIS_OUTPUT_DIR)
            print(f"已保存: {img_path}")

if __name__ == '__main__':
    main()