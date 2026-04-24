import os
import torch
import yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures
from scipy.stats import qmc
from model import Generator
from utils import calculate_relative_thickness
from foildata.xfoil import run_xfoil_single

# 默认配置
DEFAULT_N_COND = 50
DEFAULT_K_SAMPLES = 8
DEFAULT_TOP_M = 5

# Parameter Bounds: [Alpha, Re, Cl, Thickness]
BOUNDS = np.array([
    [0.0, 8.0],
    [100000.0, 600000.0],
    [-0.5, 1.75],
    [0.024, 0.198]
])

def plot_heatmap(x, y, z, title, filename, cmap='jet'):
    plt.figure(figsize=(9, 7))
    # 使用 tricontourf 生成平滑的等高线图
    # levels 增加平滑度
    levels = 20
    try:
        cntr = plt.tricontourf(x, y, z, levels=levels, cmap=cmap)
        plt.colorbar(cntr, label='Inaccuracy (Mean Error % + Variance)')
    except Exception as e:
        print(f"Warning: Could not create contour plot for {title}: {e}. Falling back to scatter.")
        sc = plt.scatter(x, y, c=z, cmap=cmap, edgecolors='k', alpha=0.8)
        plt.colorbar(sc, label='Inaccuracy (Mean Error % + Variance)')
    
    # 叠加散点显示原始采样点位置
    plt.scatter(x, y, c='k', s=10, alpha=0.4)
    
    plt.xlabel('Alpha (deg)')
    plt.ylabel('Reynolds Number')
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def _worker_eval_xfoil(args):
    """Worker function for parallel XFoil evaluation"""
    coords, re_input, alpha_input, target_cl, target_thick = args
    
    thickness = calculate_relative_thickness(coords)
    thick_err = abs(thickness - target_thick) / (target_thick + 1e-4) * 100
    
    xfoil_res = run_xfoil_single(coords, re_input, alpha_input, return_all=True)
    
    cl = xfoil_res.get('CL', np.nan) if xfoil_res else np.nan
    cd = xfoil_res.get('CD', np.nan) if xfoil_res else np.nan
    cm = xfoil_res.get('CM', np.nan) if xfoil_res else np.nan

    # 只有当 CL, CD, CM 全部有效时才计算误差，否则给予惩罚值
    if not np.isnan(cl) and not np.isnan(cd) and not np.isnan(cm):
        cl_err = abs(cl - target_cl) / (abs(target_cl) + 1e-4) * 100
    else:
        cl_err = 10.0 # Penalty for failure or partial convergence
        
    total_err = 0.5 * thick_err + 0.5 * cl_err
    
    return {
        'coords': coords, 'total_err': total_err, 'thick_err': thick_err, 'cl_err': cl_err,
        'thickness': thickness, 'cl': cl, 'cd': cd, 'cm': cm,
        'alpha': alpha_input, 're': re_input, 'target_thick': target_thick, 'target_cl': target_cl
    }

def evaluate_model(model_path, tag, config, device, cond_mean, cond_std, n_cond, k_samples, top_m):
    print(f"\n--- Evaluating {tag} model: {model_path} ---")
    if not os.path.exists(model_path):
        print(f"Model {model_path} not found. Skipping.")
        return

    generator = Generator(config).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    if isinstance(checkpoint, dict) and 'generator_state_dict' in checkpoint:
        generator.load_state_dict(checkpoint['generator_state_dict'])
    else:
        generator.load_state_dict(checkpoint)
    generator.eval()

    # Generate LHS samples
    sampler = qmc.LatinHypercube(d=4)
    sample = sampler.random(n=n_cond)
    cond_samples = qmc.scale(sample, BOUNDS[:, 0], BOUNDS[:, 1])

    noise_dim = config.get('noise_dimension', 6)
    num_output_points = config.get('num_output_points', 100)

    all_eval_tasks = []
    
    print(f"Generating {n_cond * k_samples} airfoils for evaluation...")
    for i, cond_val in enumerate(cond_samples):
        alpha_input, re_input, target_cl, target_thick = cond_val
        
        cond_tensor = torch.tensor(cond_val, dtype=torch.float32).unsqueeze(0).to(device)
        norm_cond = (cond_tensor - cond_mean) / cond_std
        norm_cond = norm_cond.expand(k_samples, -1)
        
        noise = torch.randn(k_samples, noise_dim).to(device)
        
        with torch.no_grad():
            gen_out = generator(noise, norm_cond)
        gen_airfoils = gen_out.view(k_samples, num_output_points, 2).cpu().numpy()
        
        for k in range(k_samples):
            all_eval_tasks.append((gen_airfoils[k], re_input, alpha_input, target_cl, target_thick))

    # Parallel Execution
    max_workers = config['max_workers']
    print(f"Running parallel XFoil evaluations (max_workers={max_workers})...")
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(_worker_eval_xfoil, all_eval_tasks))

    # Process results back into heatmaps data
    results_alpha, results_re = [], []
    results_thick_inacc, results_cl_inacc = [], []
    
    for i in range(n_cond):
        batch = results[i*k_samples : (i+1)*k_samples]
        batch_thick_errs = [r['thick_err'] for r in batch]
        batch_cl_errs = [r['cl_err'] for r in batch]
        
        thick_inacc = np.mean(batch_thick_errs) + np.var(batch_thick_errs)
        cl_inacc = np.mean(batch_cl_errs) + np.var(batch_cl_errs)
        
        # Take metadata from first sample of batch
        meta = batch[0]
        results_alpha.append(meta['alpha'])
        results_re.append(meta['re'])
        results_thick_inacc.append(thick_inacc)
        results_cl_inacc.append(cl_inacc)

    # Plot heatmaps
    os.makedirs('model', exist_ok=True)
    plot_heatmap(results_alpha, results_re, results_thick_inacc, f'{tag} Thickness Inaccuracy', f'model/eval_{tag.lower()}_thick.png')
    plot_heatmap(results_alpha, results_re, results_cl_inacc, f'{tag} CL Inaccuracy', f'model/eval_{tag.lower()}_cl.png')
                 
    # Save top M
    os.makedirs('foildata/gen', exist_ok=True)
    results.sort(key=lambda x: x['total_err'])
    print(f"\nSaving Top {top_m} {tag} airfoils...")
    
    for i, item in enumerate(results[:top_m]):
        filename = f"{tag}_Top{i+1}_Terr{item['thick_err']:.1f}_Clerr{item['cl_err']:.1f}_T{item['thickness']:.4f}_Cl{item['cl']:.4f}_Cd{item['cd']:.5f}.dat"
        filepath = os.path.join('foildata/gen', filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            header = f"{tag}_Top{i+1}_Terr_{item['thick_err']:.1f}_Clerr_{item['cl_err']:.1f}_Thick_{item['thickness']:.4f}_Cl_{item['cl']:.4f}_Cd_{item['cd']:.5f}"
            f.write(header + "\n")
            for pt in item['coords']:
                f.write(f"{pt[0]:.6f} {pt[1]:.6f}\n")
        print(f"Saved: {filename}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_cond", type=int, default=DEFAULT_N_COND, help="Number of LHS conditions")
    parser.add_argument("--k_samples", type=int, default=DEFAULT_K_SAMPLES, help="Airfoils per condition")
    parser.add_argument("--top_m", type=int, default=DEFAULT_TOP_M, help="Top M airfoils to save")
    args = parser.parse_args()

    with open("config.yaml", 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    device_cfg = config.get("device", "auto")
    if device_cfg.lower() == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif device_cfg.lower() == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    norm_params = torch.load('model/cond_norm.pt', map_location=device, weights_only=True)
    cond_mean = norm_params['mean'].to(device)
    cond_std = norm_params['std'].to(device)

    evaluate_model('model/pre_train.pt', 'PRE', config, device, cond_mean, cond_std, args.n_cond, args.k_samples, args.top_m)
    evaluate_model('model/gan_final.pt', 'PG', config, device, cond_mean, cond_std, args.n_cond, args.k_samples, args.top_m)
