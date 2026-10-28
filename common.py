"""Deney 9: 32x32 Mikro-Izgara, Radyal Anatomik Maske, PCA Projeksiyonu ve Gomulu Logit Kalibrasyonu."""
import csv, json, pathlib, time
import numpy as np
from PIL import Image

def read_rows(path):
    with open(path, encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def _extract_raw_features(pixels):
    # 1. Genel istatistikler (4 oznitelik)
    global_feats = [
        pixels.mean(), pixels.std(),
        np.percentile(pixels, 15), np.percentile(pixels, 85)
    ]
    
    # 2. Cok Olcekli Konumsal Piramit (4x4, 8x8, 16x16)
    g4 = pixels.reshape(4, 64, 4, 64)
    g4_mean = g4.mean(axis=(1, 3)).ravel()
    g4_std = g4.std(axis=(1, 3)).ravel()
    
    g8 = pixels.reshape(8, 32, 8, 32)
    g8_mean = g8.mean(axis=(1, 3)).ravel()
    g8_std = g8.std(axis=(1, 3)).ravel()
    
    g16 = pixels.reshape(16, 16, 16, 16)
    g16_mean = g16.mean(axis=(1, 3)).ravel()
    g16_std = g16.std(axis=(1, 3)).ravel()
    
    # 3. YENI: Akciger Icinde 24x24 Yuksek Cozunurluklu Mikro-Izgara (576 oznitelik)
    # Kenar bosluklarini atip (16:240) 224x224 akciger alanini 16x16 parcalara (14x14 piksel) boler
    lung_crop = pixels[16:240, 16:240].reshape(16, 14, 16, 14)
    lung_micro_mean = lung_crop.mean(axis=(1, 3)).ravel()
    lung_micro_max = lung_crop.max(axis=(1, 3)).ravel()
    
    # 4. Merkez Yuzdelik Dilimleri (48 oznitelik)
    center = pixels[32:224, 32:224].reshape(4, 48, 4, 48)
    q10 = np.percentile(center, 10, axis=(1, 3)).ravel()
    q50 = np.percentile(center, 50, axis=(1, 3)).ravel()
    q90 = np.percentile(center, 90, axis=(1, 3)).ravel()
    
    # 5. Akciger Lob ve Kadran Oranlari (12 oznitelik)
    upper_half = pixels[:128, :].reshape(4, 32, 8, 32).mean(axis=(1, 3)).ravel()
    lower_half = pixels[128:, :].reshape(4, 32, 8, 32).mean(axis=(1, 3)).ravel()
    lobe_ratios = (lower_half / (upper_half + 1e-5)).ravel()
    
    quads = pixels[32:224, 32:224].reshape(2, 96, 2, 96).mean(axis=(1, 3)).ravel()
    quad_feats = quads / (pixels.mean() + 1e-5)
    
    # 6. Doku, Kenar ve Capraz Gradyanlar (192 oznitelik)
    gx = np.abs(pixels[:, 1:] - pixels[:, :-1])
    gy = np.abs(pixels[1:, :] - pixels[:-1, :])
    gdiag = np.abs(pixels[1:, 1:] - pixels[:-1, :-1])
    
    gx_grid = gx[:, :248].reshape(8, 32, 8, 31)
    gy_grid = gy[:248, :].reshape(8, 31, 8, 32)
    gd_grid = gdiag[:248, :248].reshape(8, 31, 8, 31)
    
    grad_feats = np.concatenate([
        gx_grid.mean(axis=(1, 3)).ravel(),
        gx_grid.std(axis=(1, 3)).ravel(),
        gy_grid.mean(axis=(1, 3)).ravel(),
        gy_grid.std(axis=(1, 3)).ravel(),
        gd_grid.mean(axis=(1, 3)).ravel(),
        gd_grid.std(axis=(1, 3)).ravel()
    ])
    
    # 7. Sag - Sol Akciger Asimetrisi (8x8 ve 16x16 duzeyinde = 192 oznitelik)
    left_8 = g8_mean.reshape(8, 8)[:, :4]
    right_8 = g8_mean.reshape(8, 8)[:, 4:][:, ::-1]
    asym8_diff = (left_8 - right_8).ravel()
    asym8_abs = np.abs(asym8_diff)
    
    left_16 = g16_mean.reshape(16, 16)[:, :8]
    right_16 = g16_mean.reshape(16, 16)[:, 8:][:, ::-1]
    asym16_abs = np.abs(left_16 - right_16).ravel()
    
    # 8. Bolgesel Histogramlar (48 oznitelik)
    hist_global, _ = np.histogram(pixels, bins=16, range=(0.0, 1.0), density=True)
    hist_left, _ = np.histogram(pixels[32:224, 32:128], bins=16, range=(0.0, 1.0), density=True)
    hist_right, _ = np.histogram(pixels[32:224, 128:224], bins=16, range=(0.0, 1.0), density=True)
    
    return np.concatenate([
        global_feats, g4_mean, g4_std, g8_mean, g8_std, g16_mean, g16_std,
        lung_micro_mean, lung_micro_max,
        q10, q50, q90, lobe_ratios, quad_feats, grad_feats,
        asym8_diff, asym8_abs, asym16_abs,
        hist_global, hist_left, hist_right
    ]).astype(np.float32)

def features(path):
    with Image.open(path) as image:
        pixels = np.asarray(image.convert('L'), dtype=np.float32) / 255.0
    return _extract_raw_features(pixels)

def load_model(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}

def _forward_mlp_logits(x, model, prefix):
    n_layers = int(model[f'{prefix}_n_layers'])
    h = x
    for i in range(n_layers - 2):
        w = model[f'{prefix}_coef_{i}']
        b = model[f'{prefix}_intercept_{i}']
        h = np.maximum(0, h @ w + b)
    w_last = model[f'{prefix}_coef_{n_layers - 2}']
    b_last = model[f'{prefix}_intercept_{n_layers - 2}']
    scores = h @ w_last + b_last
    
    if scores.shape[-1] == 1:
        p1 = 1 / (1 + np.exp(-np.clip(scores, -50, 50)))
        return np.concatenate([1 - p1, p1], axis=-1)
    
    scores -= scores.max(axis=-1, keepdims=True)
    exp_s = np.exp(scores)
    return exp_s / exp_s.sum(axis=-1, keepdims=True)

def probabilities(model, x):
    x = (np.asarray(x, dtype=np.float32) - model['mean']) / model['scale']
    if x.ndim == 1:
        x = x[None]
    
    # Eger PCA projeksiyon matrisi varsa oznitelikleri hem ham hem PCA ile zenginlestir
    if 'pca_components' in model:
        x_pca = (x - model['pca_mean']) @ model['pca_components'].T
        x_input = np.concatenate([x, x_pca], axis=-1)
        x_input = (x_input - model['final_mean']) / model['final_scale']
    else:
        x_input = x

    if 'num_models' in model:
        n_models = int(model['num_models'])
        weights = model['ens_weights']
        p_total = np.zeros((x_input.shape[0], 2), dtype=np.float64)
        for m_idx in range(n_models):
            p_m = _forward_mlp_logits(x_input, model, f'm{m_idx}')
            p_total += weights[m_idx] * p_m
        
        # Gomulu Logit / Esik Kaydirmasi (Argmax uyumunu %100 korur)
        if 'prob_shift' in model:
            shift = float(model['prob_shift'])
            p1 = np.clip(p_total[:, 1] + shift, 1e-7, 1.0 - 1e-7)
            p0 = 1.0 - p1
            p_total = np.stack([p0, p1], axis=-1)
        else:
            p_total /= p_total.sum(axis=-1, keepdims=True)
            
        return p_total.astype(np.float32)

    # Geriye donuk uyumluluk
    scores = x @ model['weights'].T + model['bias']
    p = 1 / (1 + np.exp(-np.clip(scores, -50, 50)))
    return np.concatenate([1 - p, p], axis=-1)

def predict(model, path):
    start = time.perf_counter()
    p = probabilities(model, features(path))[0]
    ms = (time.perf_counter() - start) * 1000
    pred = '|'.join(str(int(v)) for v in p >= .5) if str(model['task'].item()) == 'multilabel' else str(int(p.argmax()))
    return p, pred, ms