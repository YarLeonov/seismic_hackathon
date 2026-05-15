import torch
import numpy as np
import glob
import os
from model import UNet3D
from tqdm import tqdm

def run_submission(test_dir, output_dir, weights_path):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Генерация сабмита на: {device}")
    
    model = UNet3D(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    
    files = sorted(glob.glob(os.path.join(test_dir, '*.npz')))
    print(f"Найдено {len(files)} файлов для предсказания...")
    
    for f in tqdm(files, desc="Submission inference"):
        data = np.load(f)
        cube = data['seis']
        
        # Нормализация
        mean = np.mean(cube)
        std = np.std(cube)
        if std > 1e-6:
             cube = (cube - mean) / std
             
        tensor_x = torch.tensor(cube, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        
        with torch.no_grad():
            preds = model(tensor_x)
            prob = torch.sigmoid(preds)
            pred_mask = (prob.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
            
        base_name = os.path.basename(f).replace('.npz', '')
        output_path = os.path.join(output_dir, f"{base_name}_predicted.npy")
        
        # Сохранение numpy массива маски для жюри
        np.save(output_path, pred_mask)
        
    print(f"✅ Успешно! Файлы сабмита сохранены в {output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Submit Generation Script")
    parser.add_argument('--test_dir', type=str, default=r"D:\DS 2026\seis_test", help="Path to hidden test data")
    parser.add_argument('--output_dir', type=str, default=r"D:\DS 2026\seismic_hackathon\submission", help="Path to save predictions")
    parser.add_argument('--weights', type=str, default="../unet3d_epoch_9.pt", help="Path to best model weights")
    args = parser.parse_args()
    
    # Для теста можно поменять test_dir на папку с кубами
    if os.path.exists(args.test_dir):
        run_submission(args.test_dir, args.output_dir, args.weights)
    else:
        print(f"Директория {args.test_dir} не найдена. Создайте папку с тестовыми кубами.")
