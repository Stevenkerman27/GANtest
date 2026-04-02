import torch
import yaml
import matplotlib.pyplot as plt
from model import BezierDecoderLayer
from encode_dat import load_dat
import os

def main():
    if not os.path.exists('config.yaml'):
        print("config.yaml not found.")
        return
        
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    target_path = "foildata/coord_seligFmt/naca2408.dat"
    target_points = load_dat(target_path).numpy()
    
    model = BezierDecoderLayer('config.yaml')
    
    out_file = "encoded_bezier.pt"
    if not os.path.exists(out_file):
        print(f"Encoded file {out_file} not found. Run encode_dat.py first.")
        return
        
    data = torch.load(out_file)
    control_points = data['control_points']
    weights = data['weights']
    
    with torch.no_grad():
        curve_points = model(control_points, weights).squeeze(0).numpy()
        
    cp_np = control_points.squeeze(0).numpy()
    
    plt.figure(figsize=(10, 4))
    plt.plot(target_points[:, 0], target_points[:, 1], 'k.', label='Original .dat', markersize=3)
    plt.plot(curve_points[:, 0], curve_points[:, 1], 'r-', label='Bezier Curve', linewidth=2)
    plt.plot(cp_np[:, 0], cp_np[:, 1], 'gx--', label='Control Points', alpha=0.6, markersize=5)
    plt.axis('equal')
    plt.legend()
    plt.title(f"Airfoil Bezier Encoding ({config['num_control_points']} CP, {config['num_output_points']} Pts)")
    plt.grid(True, linestyle='--', alpha=0.5)
    
    res_file = 'bezier_fit_result.png'
    plt.savefig(res_file, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {res_file}")
    # Comment out plt.show() for CLI script execution without blocking
    # plt.show()

if __name__ == '__main__':
    main()
