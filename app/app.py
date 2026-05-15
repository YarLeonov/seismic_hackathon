import streamlit as st
import numpy as np
import plotly.graph_objects as go
import torch
import sys
import os

# Подключаем модули базлайна
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from baseline.model import UNet3D

st.set_page_config(page_title="Seismic Fault Detection AI", layout="wide")
st.title("Выделение геологических разломов 3D (Real Inference)")

@st.cache_resource
def load_model(weights_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet3D(in_channels=1, out_channels=1)
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device

with st.sidebar:
    st.header("Настройки")
    
    # Ищем все .pt веса в папке проекта
    weights_dir = os.path.join(os.path.dirname(__file__), "..")
    import glob
    available_weights = sorted(glob.glob(os.path.join(weights_dir, "*.pt")))
    weight_names = [os.path.basename(w) for w in available_weights]
    
    if weight_names:
        # Устанавливаем последнюю эпоху по умолчанию (или можно выбрать любую другую)
        selected_weight = st.selectbox("Веса модели", weight_names, index=len(weight_names)-1)
        weights_file = os.path.join(weights_dir, selected_weight)
    else:
        weights_cwd = os.path.join(weights_dir, "unet3d_epoch_9.pt")
        weights_file = st.text_input("Путь к весам модели", value=weights_cwd)
        
    uploaded_file = st.file_uploader("Загрузить куб (.npz)", type=['npz'])
    run_inference_btn = st.button("Сделать предсказание (Inference)", type="primary")

cube = None
mask_gt = None

if uploaded_file is not None:
    data = np.load(uploaded_file)
    cube = data['seis']
    if 'fault' in data:
        mask_gt = data['fault']

if cube is not None and run_inference_btn:
    model, device = load_model(weights_file)
    with st.spinner("Прогон 3D куба через U-Net..."):
        # Локальная Z-score нормализация
        mean = np.mean(cube)
        std = np.std(cube)
        cube_norm = (cube - mean) / (std + 1e-6)
        
        input_tensor = torch.tensor(cube_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        
        with torch.no_grad():
            preds = model(input_tensor)
            preds_prob = torch.sigmoid(preds)
            
        pred_mask = (preds_prob.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
        st.session_state['pred_mask'] = pred_mask
    st.success("Инференс завершен!")

if 'pred_mask' in st.session_state:
    pred_mask = st.session_state['pred_mask']
else:
    pred_mask = None

if cube is not None:
    st.subheader("Анализ срезов (Slicer)")
    d, h, w = cube.shape
    slice_type = st.radio("Выбор плоскости:", ["In-line (Y)", "Cross-line (X)", "Depth/Time (Z)"], horizontal=True)

    idx_max = h-1 if "In-line" in slice_type else w-1 if "Cross-line" in slice_type else d-1
    idx = st.slider("Номер среза:", 0, idx_max, idx_max//2)

    if "In-line" in slice_type:
        img = cube[:, idx, :]
        img_gt = mask_gt[:, idx, :] if mask_gt is not None else None
        img_pred = pred_mask[:, idx, :] if pred_mask is not None else None
    elif "Cross-line" in slice_type:
        img = cube[:, :, idx]
        img_gt = mask_gt[:, :, idx] if mask_gt is not None else None
        img_pred = pred_mask[:, :, idx] if pred_mask is not None else None
    else:
        img = cube[idx, :, :]
        img_gt = mask_gt[idx, :, :] if mask_gt is not None else None
        img_pred = pred_mask[idx, :, :] if pred_mask is not None else None

    cols = st.columns(3 if mask_gt is not None else 2)
    
    with cols[0]:
        st.markdown("**Оригинальная сейсмика**")
        fig = go.Figure(data=go.Heatmap(z=img, colorscale='Greys', showscale=False))
        fig.update_layout(height=400, margin=dict(l=0, r=0, t=0, b=0))
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, width='stretch', key="plot_seis")
        
    with cols[1]:
        st.markdown("**Предсказание (Нейросеть)**")
        fig = go.Figure(data=go.Heatmap(z=img, colorscale='Greys', showscale=False))
        if img_pred is not None:
            mask_to_plot = np.where(img_pred > 0, img_pred, np.nan) 
            fig.add_trace(go.Heatmap(z=mask_to_plot, colorscale='Reds', opacity=0.5, showscale=False))
        fig.update_layout(height=400, margin=dict(l=0, r=0, t=0, b=0))
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, width='stretch', key="plot_pred")

    if mask_gt is not None and len(cols) == 3:
        with cols[2]:
            st.markdown("**Ground Truth (Разметка эксперта)**")
            fig = go.Figure(data=go.Heatmap(z=img, colorscale='Greys', showscale=False))
            mask_to_plot_gt = np.where(img_gt > 0, img_gt, np.nan) 
            fig.add_trace(go.Heatmap(z=mask_to_plot_gt, colorscale='Blues', opacity=0.5, showscale=False))
            fig.update_layout(height=400, margin=dict(l=0, r=0, t=0, b=0))
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(fig, width='stretch', key="plot_gt")
