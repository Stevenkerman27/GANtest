import torch
from torch.utils.data import Dataset

class AirfoilDataset(Dataset):
    def __init__(self, data_path, cond_norm_path="model/cond_norm.pt", coord_norm_path="model/coord_norm.pt"):
        raw_data = torch.load(data_path)
        
        # 1. Normalize conditions (labels)
        y_all = torch.stack([item['y'] for item in raw_data])
        self.y_mean = y_all.mean(dim=0)
        self.y_std = y_all.std(dim=0) + 1e-8
        
        # Save condition normalization parameters
        torch.save({'mean': self.y_mean, 'std': self.y_std}, cond_norm_path)
        
        # 2. Normalize airfoil y-coordinates (features)
        coords_all = torch.stack([item['x'] for item in raw_data]) # (N, M*2)
        coords_all = coords_all.view(coords_all.size(0), -1, 2)
        y_coords = coords_all[:, :, 1]
        self.coord_y_mean = y_coords.mean()
        self.coord_y_std = y_coords.std() + 1e-8
        
        # Save coordinate normalization parameters
        torch.save({'mean': self.coord_y_mean, 'std': self.coord_y_std}, coord_norm_path)
        
        self.data = []
        for item in raw_data:
            # Normalize conditions
            norm_y = (item['y'] - self.y_mean) / self.y_std
            
            # Normalize only y-coordinates in x feature
            c = item['x'].clone().view(-1, 2)
            c[:, 1] = (c[:, 1] - self.coord_y_mean) / self.coord_y_std
            norm_x = c.view(-1)
            
            self.data.append({
                'x': norm_x,
                'y': norm_y
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]['x'], self.data[idx]['y']