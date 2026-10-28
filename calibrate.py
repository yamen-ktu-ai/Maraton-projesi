"""Ozgün Katki: Sicaklik Kalibrasyonu (Temperature Scaling) ve Guven Analizi."""
import argparse, json, pathlib
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import log_loss, brier_score_loss
from common import load_model, probabilities, features, read_rows

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', default='calismalar/calibration_report.json')
    return parser.parse_args()

def expected_calibration_error(y_true, y_prob, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_mask = (y_prob > bin_boundaries[i]) & (y_prob <= bin_boundaries[i + 1])
        bin_size = np.sum(bin_mask)
        if bin_size > 0:
            bin_acc = np.mean(y_true[bin_mask])
            bin_conf = np.mean(y_prob[bin_mask])
            ece += (bin_size / len(y_true)) * np.abs(bin_acc - bin_conf)
    return float(ece)

def main():
    args = parse_args()
    data_dir = pathlib.Path(args.data)
    model = load_model(args.model)
    
    val_rows = read_rows(data_dir / 'val.csv')
    print(f"Dogrulama kumesi icin kalibrasyon verisi toplaniyor ({len(val_rows)} ornek)...")
    
    y_true = []
    raw_probs = []
    
    for r in val_rows:
        p_path = data_dir / r['path']
        p = probabilities(model, features(p_path))[0]
        raw_probs.append(p[1])
        y_true.append(int(r['label']))
        
    y_true = np.array(y_true, dtype=np.int64)
    raw_probs = np.clip(np.array(raw_probs, dtype=np.float64), 1e-7, 1 - 1e-7)
    
    # Ham logitleri hesapla: log(p / (1-p))
    logits = np.log(raw_probs / (1.0 - raw_probs))
    
    # NLL minimize eden T (Temperature) optimizasyonu
    def cost_func(t_val):
        t = t_val[0]
        scaled_logits = logits / t
        p_scaled = 1.0 / (1.0 + np.exp(-np.clip(scaled_logits, -50, 50)))
        return log_loss(y_true, p_scaled)
    
    res = minimize(cost_func, x0=[1.0], bounds=[(0.05, 5.0)], method='L-BFGS-B')
    optimal_t = float(res.x[0])
    
    calibrated_probs = 1.0 / (1.0 + np.exp(-np.clip(logits / optimal_t, -50, 50)))
    
    # Metrik karsilastirmasi
    raw_ece = expected_calibration_error(y_true, raw_probs)
    cal_ece = expected_calibration_error(y_true, calibrated_probs)
    raw_brier = float(brier_score_loss(y_true, raw_probs))
    cal_brier = float(brier_score_loss(y_true, calibrated_probs))
    
    # Gri bolge analizi (0.40 - 0.60 araligi)
    uncertain_mask = (calibrated_probs >= 0.40) & (calibrated_probs <= 0.60)
    uncertain_count = int(np.sum(uncertain_mask))
    uncertain_ratio = float(uncertain_count / len(y_true))
    
    report = {
        "optimal_temperature": round(optimal_t, 4),
        "raw_metrics": {
            "expected_calibration_error_ECE": round(raw_ece, 5),
            "brier_score": round(raw_brier, 5),
            "log_loss": round(float(log_loss(y_true, raw_probs)), 5)
        },
        "calibrated_metrics": {
            "expected_calibration_error_ECE": round(cal_ece, 5),
            "brier_score": round(cal_brier, 5),
            "log_loss": round(float(log_loss(y_true, calibrated_probs)), 5)
        },
        "clinical_decision_support": {
            "uncertain_zone_bounds": [0.40, 0.60],
            "flagged_uncertain_cases": uncertain_count,
            "uncertain_cases_percentage": f"{uncertain_ratio * 100:.2f}%",
            "clinical_advice": "Gri bolgedeki vakalar icin yapay zeka karari yerine hekim klinik inceleme uyarisi uretilir."
        }
    }
    
    out_p = pathlib.Path(args.output)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        
    print("\n==========================================")
    print(f"Optimal Sicaklik (T): {optimal_t:.4f}")
    print(f"Kalibrasyon Oncesi ECE: {raw_ece:.4f} -> Kalibrasyon Sonrasi ECE: {cal_ece:.4f}")
    print(f"Brier Skoru Kazanci: {raw_brier:.4f} -> {cal_brier:.4f}")
    print(f"Gri Bolge (Belirsiz) Vaka Orani: %{uncertain_ratio * 100:.2f} ({uncertain_count} vaka)")
    print(f"Rapor Kaydedildi: {args.output}")
    print("==========================================")

if __name__ == '__main__':
    main()