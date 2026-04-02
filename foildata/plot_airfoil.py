import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import re

# 尝试导入 windnd 用于拖拽支持
try:
    import windnd
except ImportError:
    print("Error: 'windnd' library not found. Please install it using: pip install windnd")
    windnd = None

class AirfoilPlotter:
    def __init__(self, root):
        self.root = root
        self.root.title("翼型可视化工具 (拖入 .dat 文件)")
        self.root.geometry("1200x900")

        # 初始化绘图区
        self.fig, self.ax = plt.subplots(figsize=(8, 5))
        self.ax.set_aspect('equal')
        self.ax.grid(True, linestyle='--', alpha=0.6)
        self.ax.set_xlabel("X/c")
        self.ax.set_ylabel("Y/c")
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 提示标签
        self.label = tk.Label(self.root, text="请将 .dat 翼型文件拖入此窗口", font=("Microsoft YaHei", 12), pady=10)
        self.label.pack()

        # 注册拖拽事件
        if windnd:
            windnd.hook_dropfiles(self.root, func=self.on_drop)
        else:
            messagebox.showwarning("依赖缺失", "未检测到 windnd 库，拖拽功能将不可用。\n请运行: pip install windnd")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def on_closing(self):
        self.root.destroy()
        import sys
        sys.exit(0)  # 显式告诉系统：我彻底走人了

    def parse_selig(self, file_path):
        """解析 Selig 格式的 .dat 文件"""
        coords = []
        name = "Unknown Airfoil"
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                if not lines:
                    return None, None
                
                # 第一行通常是名称
                name = lines[0].strip()
                
                # 正则匹配浮点数对
                coord_pattern = re.compile(r"^\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$")
                
                for line in lines[1:]:
                    match = coord_pattern.match(line)
                    if match:
                        x = float(match.group(1))
                        y = float(match.group(2))
                        coords.append((x, y))
            
            if not coords:
                return None, name
                
            return coords, name
        except Exception as e:
            print(f"解析错误: {e}")
            return None, name

    def on_drop(self, files):
        """处理文件拖入"""
        for file_path in files:
            # 转换字节流路径为字符串 (windnd 在某些版本返回 bytes)
            if isinstance(file_path, bytes):
                file_path = file_path.decode('gbk') # Windows 系统中文路径常用 GBK
            
            if not file_path.lower().endswith('.dat'):
                continue
                
            coords, name = self.parse_selig(file_path)
            
            if coords:
                self.draw_airfoil(coords, name, os.path.basename(file_path))
            else:
                messagebox.showerror("格式错误", f"无法解析文件: {os.path.basename(file_path)}\n请确保它是 Selig 格式的坐标文件。")
            break # 每次只画第一个拖入的文件

    def draw_airfoil(self, coords, name, filename):
        """绘制翼型"""
        self.ax.clear()
        self.ax.set_aspect('equal')
        self.ax.grid(True, linestyle='--', alpha=0.6)
        
        xs, ys = zip(*coords)
        
        # 绘图
        self.ax.plot(xs, ys, 'b-', linewidth=1.5, label=f"{name}")
        self.ax.scatter(xs, ys, s=5, c='red', alpha=0.5) # 显示点
        
        self.ax.set_title(f"Airfoil: {name} ({filename})", fontsize=12)
        self.ax.set_xlabel("X/c")
        self.ax.set_ylabel("Y/c")
        
        # 自动调整范围
        self.ax.set_xlim(-0.05, 1.05)
        # 动态调整 Y 轴比例
        y_max = max(abs(min(ys)), abs(max(ys)))
        self.ax.set_ylim(-y_max * 1.5 - 0.1, y_max * 1.5 + 0.1)
        
        self.canvas.draw()
        self.label.config(text=f"当前显示: {filename}")

if __name__ == "__main__":
    root = tk.Tk()
    app = AirfoilPlotter(root)
    root.mainloop()
