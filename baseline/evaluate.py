import torch
import numpy as np
import glob
import os
from sklearn.metrics import f1_score
from model import UNet3D
from tqdm import tqdm

def evaluate_model(data_dir, weights_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используемое устройство: {device}")
    
    # Инициализируем и загружаем веса
    model = UNet3D(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    
    # Находим все файлы (для валидации обычно берут сплит, здесь мы просто возьмем 10 последних)
    files = sorted(glob.glob(os.path.join(data_dir, '*.npz')))
    val_files = files[-10:] if len(files) > 10 else files
    
    all_preds = []
    all_targets = []
    
    print(f"Оценка F1-Score на {len(val_files)} кубах из валидационной выборки...")
    
    for f in tqdm(val_files, desc="Evaluation"):
        data = np.load(f)
        cube = data['seis']
        target = data['fault']
        
        # Z-score нормализация, как при обучении
        mean = np.mean(cube)
        std = np.std(cube)
        if std > 1e-6:
             cube = (cube - mean) / std
             
        tensor_x = torch.tensor(cube, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        
        with torch.no_grad():
            preds = model(tensor_x)
            prob = torch.sigmoid(preds)
            # Биннаризация по порогу 0.5
            pred_mask = (prob.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
            
        all_preds.append(pred_mask.flatten())
        all_targets.append(target.flatten())
        
    all_preds_concat = np.concatenate(all_preds)
    all_targets_concat = np.concatenate(all_targets)
    
    # F1 score (average='binary' т.к. классы 0 и 1)
    f1 = f1_score(all_targets_concat, all_preds_concat, average='binary')
    print(f"🎯 Итоговый F1-Score (Validation): {f1:.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluation Script")
    parser.add_argument('--data_dir', type=str, default=r"D:\DS 2026\seis", help="Path to data")
    parser.add_argument('--weights', type=str, default="../unet3d_epoch_9.pt", help="Path to best model weights")
    args = parser.parse_args()
    
    evaluate_model(args.data_dir, args.weights)
