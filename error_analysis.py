"""Zorunlu Cikti: En az 5 hatali ornegin gorseli ve istatistiksel incelemesi (Yuksek Cozunurluklu PNG)."""
import json, pathlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from common import read_rows

def main():
    d = pathlib.Path('veri')
    truth = read_rows(d / 'val.csv')
    pred = read_rows('calismalar/predictions_val_deney2.csv')
    mapped = {r['id']: r for r in pred}

    fp_list = []  # Gercekte 0 (Normal), Model 1 (Opasite) demis
    fn_list = []  # Gercekte 1 (Opasite), Model 0 (Normal) demis

    for r in truth:
        ident = r['id']
        y_true = int(r['label'])
        p_row = mapped[ident]
        y_pred = int(p_row['prediction'])
        prob_0 = float(p_row['prob_0'])
        prob_1 = float(p_row['prob_1'])

        if y_true != y_pred:
            img_path = d / r['path']
            with Image.open(img_path) as im:
                arr = np.asarray(im.convert('L'), dtype=np.float32) / 255.0
                mean_val = float(arr.mean())
                std_val = float(arr.std())
            info = {
                'id': ident,
                'path': r['path'],
                'true': y_true,
                'pred': y_pred,
                'prob_0': prob_0,
                'prob_1': prob_1,
                'conf': max(prob_0, prob_1),
                'mean': mean_val,
                'std': std_val
            }
            if y_true == 0 and y_pred == 1:
                fp_list.append(info)
            else:
                fn_list.append(info)

    fp_list.sort(key=lambda x: x['prob_1'], reverse=True)
    fn_list.sort(key=lambda x: x['prob_0'], reverse=True)

    # 3 adet False Negative ve 3 adet False Positive (Toplam 6 örnek)
    selected = fn_list[:3] + fp_list[:3]

    # Çözünürlüğü artırmak için görsel boyutunu 512x512 yapıyoruz
    w, h = 512, 512
    pad_top = 110
    margin = 25
    spacing = 25
    
    canvas_w = w * 3 + margin * 4
    canvas_h = (h + pad_top) * 2 + spacing * 3
    
    canvas = Image.new('RGB', (canvas_w, canvas_h), color=(255, 255, 255))
    
    # Yazı tipi boyutu (Sistemde Arial varsa kullanır, yoksa varsayılan)
    try:
        font_title = ImageFont.truetype("arial.ttf", 22)
        font_sub = ImageFont.truetype("arial.ttf", 18)
    except IOError:
        font_title = ImageFont.load_default()
        font_sub = ImageFont.load_default()

    draw = ImageDraw.Draw(canvas)

    for idx, item in enumerate(selected):
        col = idx % 3
        row = idx // 3
        x0 = margin + col * (w + margin)
        y0 = spacing + row * (h + pad_top + spacing)

        # Görüntüyü yüksek kaliteli resampling (LANCZOS) ile büyütüyoruz
        with Image.open(d / item['path']) as im:
            im_resized = im.convert('RGB').resize((w, h), Image.Resampling.LANCZOS)
            canvas.paste(im_resized, (x0, y0 + pad_top))

        err_type = "FN (Kaçırılan Vaka)" if item['true'] == 1 else "FP (Yanlış Alarm)"
        text1 = f"Örnek #{idx+1} | {err_type}"
        text2 = f"Gerçek: {item['true']} | Tahmin: {item['pred']} (Skor: {item['prob_1']:.2f})"
        text3 = f"Ort. Parlaklık: {item['mean']:.2f} | Std: {item['std']:.2f}"

        draw.text((x0 + 8, y0 + 12), text1, fill=(180, 0, 0) if item['true'] == 1 else (0, 0, 180), font=font_title)
        draw.text((x0 + 8, y0 + 44), text2, fill=(0, 0, 0), font=font_sub)
        draw.text((x0 + 8, y0 + 74), text3, fill=(80, 80, 80), font=font_sub)

    # Kalite kaybı olmaması için PNG formatında kaydediyoruz
    out_img = pathlib.Path('calismalar/hata_analizi_6_ornek.png')
    canvas.save(out_img, 'PNG')
    print("Yüksek çözünürlüklü hata analizi görseli başarıyla oluşturuldu!")

if __name__ == '__main__':
    main()