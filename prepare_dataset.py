import os
import glob
import torch
import numpy as np

def prepare_dataset(processed_foil_dir, polars_dir, output_file="airfoil_dataset.pt"):
    """
    读取翼型坐标文件和极曲线数据，将坐标点、雷诺数（Re）、迎角（alpha）拼接为特征张量，
    并提取对应的气动参数（CL, CD, CDp, CM）作为标签，最后将整个数据集打包为列表保存为 .pt 文件。
    """
    data_list = []
    
    # 获取所有的 polar.txt 文件
    polar_files = glob.glob(os.path.join(polars_dir, "*_polar.txt"))
    
    for p_file in polar_files:
        basename = os.path.basename(p_file)
        # 解析文件名以获取翼型名称和雷诺数
        # 例如: e422_Re100000_polar.txt
        parts = basename.split('_Re')
        if len(parts) != 2:
            continue
        foil_name = parts[0]
        
        re_part = parts[1].replace('_polar.txt', '')
        try:
            Re = float(re_part)
        except ValueError:
            continue
            
        # 寻找对应的坐标文件
        foil_path = os.path.join(processed_foil_dir, f"{foil_name}.dat")
        if not os.path.exists(foil_path):
            # 处理部分带有下划线前缀的坐标文件
            foil_path = os.path.join(processed_foil_dir, f"_{foil_name}.dat")
            if not os.path.exists(foil_path):
                print(f"警告: 找不到翼型 {foil_name} 的坐标数据，跳过该文件 ({basename})。")
                continue
                
        # 读取翼型坐标 (跳过第一行标题)
        try:
            coords = np.loadtxt(foil_path, skiprows=1)
        except Exception as e:
            print(f"读取坐标文件 {foil_path} 失败: {e}")
            continue
            
        # 读取 polar 数据
        try:
            with open(p_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except UnicodeDecodeError:
            with open(p_file, 'r', encoding='latin-1') as f:
                lines = f.readlines()
                
        # 寻找数据开始的行（在 '------' 之后）
        start_idx = -1
        for i, line in enumerate(lines):
            if '------' in line:
                start_idx = i + 1
                break
                
        if start_idx == -1:
            print(f"警告: 在 {basename} 中找不到表格数据起始行。")
            continue
            
        # 解析每一行的迎角和气动参数
        for line in lines[start_idx:]:
            if not line.strip():
                continue
            vals = line.split()
            # 通常至少包含 alpha, CL, CD, CDp, CM 等
            if len(vals) < 5:
                continue
                
            try:
                alpha = float(vals[0])
                CL = float(vals[1])
                CD = float(vals[2])
                CDp = float(vals[3])
                CM = float(vals[4])
            except ValueError:
                continue
                
            N = coords.shape[0]
            
            # 方法1: 展平拼接为 1D 张量，形状为 (2N + 2,) - 适合全连接网络 (MLP)
            coords_flat = coords.flatten()
            x_1d = np.concatenate([coords_flat, [Re, alpha]])
            x_1d_tensor = torch.tensor(x_1d, dtype=torch.float32)
            
            # 方法2: 逐点拼接为 2D 张量，形状为 (N, 4) - 适合卷积或图神经网络 (CNN/GNN)
            Re_col = np.full((N, 1), Re)
            alpha_col = np.full((N, 1), alpha)
            x_2d = np.hstack((coords, Re_col, alpha_col))
            x_2d_tensor = torch.tensor(x_2d, dtype=torch.float32)
            
            # 标签：提取 CL, CD, CDp, CM (可根据需要修改)
            y_tensor = torch.tensor([CL, CD, CDp, CM], dtype=torch.float32)
            
            # 保存到数据集列表中
            data_list.append({
                "x_1d": x_1d_tensor,      # 展平的特征 [x1, y1, x2, y2, ..., Re, alpha]
                "x_2d": x_2d_tensor,      # 点云特征矩阵 [[x1, y1, Re, alpha], ...]
                "coords": torch.tensor(coords, dtype=torch.float32), # 纯坐标 (N, 2)
                "Re": torch.tensor([Re], dtype=torch.float32),
                "alpha": torch.tensor([alpha], dtype=torch.float32),
                "y": y_tensor,            # [CL, CD, CDp, CM]
                "foil_name": foil_name
            })
            
    print(f"\n数据集准备完成！总共收集了 {len(data_list)} 个样本。")
    print(f"正在保存至 {output_file} ...")
    torch.save(data_list, output_file)
    print("保存成功！可以使用 torch.load('{}') 进行读取。".format(output_file))

if __name__ == '__main__':
    # 按照当前目录结构设置路径
    processed_dir = os.path.join("foildata", "processed_foil")
    polars_dir = os.path.join("foildata", "polars")
    out_file = "airfoil_dataset.pt"
    
    prepare_dataset(processed_dir, polars_dir, out_file)
