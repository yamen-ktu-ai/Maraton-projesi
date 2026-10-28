"""Deney 9: Bellek Optimizasyonlu 4'lu Derin Komite Egitimi."""
import argparse, json, pathlib, gc
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from common import features, read_rows

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=20261023)
    return parser.parse_args()

def extract_features_and_labels(rows, data_dir):
    x_list = []
    y_list = []
    for r in rows:
        p = pathlib.Path(data_dir) / r['path']
        x_list.append(features(p))
        y_list.append(int(r['label']))
    return np.array(x_list, dtype=np.float32), np.array(y_list, dtype=np.int64)

def main():
    args = parse_args()
    data_dir = pathlib.Path(args.data)
    out_dir = pathlib.Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(data_dir / 'dataset_card.json', encoding='utf-8') as f:
        card = json.load(f)

    train_rows = read_rows(data_dir / 'train.csv')
    val_rows = read_rows(data_dir / 'val.csv')

    print(f"Oznitelikler cikariliyor ({len(train_rows)} egitim, {len(val_rows)} dogrulama)...")
    x_train, y_train = extract_features_and_labels(train_rows, data_dir)
    x_val, y_val = extract_features_and_labels(val_rows, data_dir)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train).astype(np.float32)
    x_val = scaler.transform(x_val).astype(np.float32)

    configs = [
        {"name": "M1 (384-192-96)", "layers": (384, 192, 96), "alpha": 0.003, "lr": 0.0004, "seed": args.seed},
        {"name": "M2 (256-128)",    "layers": (256, 128),     "alpha": 0.005, "lr": 0.0005, "seed": 42},
        {"name": "M3 (320-160)",    "layers": (320, 160),     "alpha": 0.004, "lr": 0.0004, "seed": 2026},
        {"name": "M4 (256-128-64)", "layers": (256, 128, 64), "alpha": 0.003, "lr": 0.0004, "seed": 999},
    ]

    models = []
    val_probs_list = []

    for idx, cfg in enumerate(configs):
        print(f"--- [{idx+1}/4] {cfg['name']} Egitiliyor ---")
        clf = MLPClassifier(
            hidden_layer_sizes=cfg["layers"],
            activation='relu',
            solver='adam',
            alpha=cfg["alpha"],
            batch_size=128, # Bellek rahatlamasi icin batch 128
            learning_rate='adaptive',
            learning_rate_init=cfg["lr"],
            max_iter=150,
            random_state=cfg["seed"],
            early_stopping=True,
            n_iter_no_change=12,
            tol=1e-5,
            verbose=False
        )
        clf.fit(x_train, y_train)
        probs = clf.predict_proba(x_val).astype(np.float32)
        score = f1_score(y_val, probs.argmax(axis=1), average='macro')
        print(f"    -> Tekil Macro-F1: {score:.6f}")
        models.append(clf)
        val_probs_list.append(probs)
        gc.collect()

    # Optimal Agirlik Aramasi
    print("\nOptimal komite agirliklari araniyor...")
    rng = np.random.default_rng(args.seed)
    best_weights = np.ones(len(models)) / len(models)
    p_init = sum(w * p for w, p in zip(best_weights, val_probs_list))
    best_f1 = f1_score(y_val, p_init.argmax(axis=1), average='macro')

    for _ in range(3000):
        w = rng.dirichlet(np.ones(len(models)))
        p_blend = sum(wi * pi for wi, pi in zip(w, val_probs_list))
        score = f1_score(y_val, p_blend.argmax(axis=1), average='macro')
        if score > best_f1:
            best_f1 = float(score)
            best_weights = w

    print(f"\n==========================================")
    print(f"Optimal Agirliklar: {[round(float(x), 3) for x in best_weights]}")
    print(f"Komite Dogrulama Macro-F1: {best_f1:.6f}")
    print(f"==========================================")

    save_dict = {
        'mean': scaler.mean_.astype(np.float32),
        'scale': scaler.scale_.astype(np.float32),
        'task': np.array(card.get('task', 'binary')),
        'classes': models[0].classes_,
        'num_models': np.array(len(models)),
        'ens_weights': np.array(best_weights, dtype=np.float32)
    }

    for m_idx, clf in enumerate(models):
        save_dict[f'm{m_idx}_n_layers'] = clf.n_layers_
        for i, (w, b) in enumerate(zip(clf.coefs_, clf.intercepts_)):
            save_dict[f'm{m_idx}_coef_{i}'] = w.astype(np.float32)
            save_dict[f'm{m_idx}_intercept_{i}'] = b.astype(np.float32)

    np.savez_compressed(out_dir / 'model.npz', **save_dict)

    summary = {
        "track": card.get("track", "T01"),
        "macro_f1": float(best_f1),
        "weights": [float(x) for x in best_weights],
        "train_count": len(train_rows),
        "val_count": len(val_rows),
        "seed": args.seed,
        "note": "Deney 9: Memory-Optimized 4-MLP Committee Ensemble"
    }
    with open(out_dir / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

if __name__ == '__main__':
    main()