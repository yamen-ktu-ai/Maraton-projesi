"""KTU.AI Marathon T01: Ultra-Luks Klinik Karar Destek ve Radyoloji Paneli."""
import argparse, base64, http.server, io, json, pathlib, socketserver, time, traceback
import numpy as np
from PIL import Image
from common import load_model, probabilities, features

HTML_PAGE = """<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>KTU.AI T01 - Radyoloji Karar Destek Kokpiti</title>
<style>
  :root {
    --bg: #090d16;
    --card: rgba(18, 26, 43, 0.75);
    --border: rgba(56, 189, 248, 0.15);
    --border-hover: rgba(56, 189, 248, 0.45);
    --accent: #38bdf8;
    --accent-glow: rgba(56, 189, 248, 0.35);
    --danger: #f43f5e;
    --danger-glow: rgba(244, 63, 94, 0.35);
    --success: #10b981;
    --success-glow: rgba(16, 185, 129, 0.35);
    --warning: #f59e0b;
    --text: #f8fafc;
    --muted: #94a3b8;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
  
  body {
    background-color: var(--bg);
    background-image: 
      radial-gradient(at 0% 0%, rgba(56, 189, 248, 0.08) 0px, transparent 50%),
      radial-gradient(at 100% 100%, rgba(16, 185, 129, 0.05) 0px, transparent 50%),
      radial-gradient(at 50% 50%, rgba(15, 23, 42, 0.6) 0px, transparent 100%);
    color: var(--text);
    min-height: 100vh;
    padding: 40px 20px;
    display: flex;
    justify-content: center;
  }

  .container { max-width: 1050px; width: 100%; }

  header { text-align: center; margin-bottom: 35px; position: relative; }
  .pulse-indicator {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(56, 189, 248, 0.1);
    border: 1px solid var(--border);
    padding: 6px 14px;
    border-radius: 30px;
    font-size: 12px;
    color: var(--accent);
    margin-bottom: 12px;
    letter-spacing: 0.5px;
    text-transform: uppercase;
  }
  .dot {
    width: 8px; height: 8px;
    background: var(--accent);
    border-radius: 50%;
    box-shadow: 0 0 10px var(--accent);
    animation: pulse 1.8s infinite;
  }
  @keyframes pulse {
    0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(56, 189, 248, 0.7); }
    70% { transform: scale(1); box-shadow: 0 0 0 8px rgba(56, 189, 248, 0); }
    100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(56, 189, 248, 0); }
  }

  header h1 {
    font-size: 30px;
    font-weight: 800;
    letter-spacing: -0.5px;
    background: linear-gradient(135deg, #ffffff 40%, var(--accent) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 8px;
  }
  header p { color: var(--muted); font-size: 14px; }

  .grid { display: grid; grid-template-columns: 1.1fr 1fr; gap: 24px; }
  @media(max-width: 860px) { .grid { grid-template-columns: 1fr; } }

  .glass-panel {
    background: var(--card);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 24px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
    transition: border 0.3s ease;
  }
  .glass-panel:hover { border-color: var(--border-hover); }

  .panel-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid var(--border);
    padding-bottom: 12px;
    margin-bottom: 18px;
  }
  .panel-header h2 { font-size: 15px; color: #cbd5e1; font-weight: 600; }
  .badge-icon { font-size: 12px; color: var(--muted); background: rgba(255,255,255,0.05); padding: 4px 8px; border-radius: 6px; }

  .upload-box {
    border: 2px dashed rgba(56, 189, 248, 0.25);
    background: rgba(56, 189, 248, 0.02);
    border-radius: 12px;
    padding: 30px 20px;
    text-align: center;
    cursor: pointer;
    transition: all 0.25s ease;
  }
  .upload-box:hover {
    border-color: var(--accent);
    background: rgba(56, 189, 248, 0.06);
    transform: translateY(-2px);
  }
  input[type="file"] { display: none; }

  .viewer-container {
    position: relative;
    width: 100%;
    margin-top: 18px;
    border-radius: 10px;
    overflow: hidden;
    background: #000;
    display: none;
    border: 1px solid rgba(255,255,255,0.1);
  }
  #preview {
    width: 100%;
    max-height: 340px;
    object-fit: contain;
    display: block;
    margin: 0 auto;
  }
  
  /* Lazer Röntgen Tarama Çizgisi */
  .scanner-line {
    position: absolute;
    top: 0; left: 0; width: 100%; height: 3px;
    background: linear-gradient(90deg, transparent, var(--accent), transparent);
    box-shadow: 0 0 15px var(--accent);
    display: none;
    animation: scanAnim 2.2s ease-in-out infinite alternate;
  }
  @keyframes scanAnim {
    0% { top: 0%; opacity: 0.8; }
    100% { top: 98%; opacity: 0.8; }
  }

  .status-placeholder {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 250px;
    color: var(--muted);
    font-size: 14px;
    text-align: center;
    gap: 12px;
  }

  .result-wrapper { display: none; animation: fadeIn 0.4s ease; }
  @keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }

  .diagnosis-badge {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    padding: 8px 16px;
    border-radius: 30px;
    font-size: 14px;
    font-weight: 700;
    letter-spacing: 0.5px;
    margin-bottom: 20px;
  }
  .badge-normal { background: rgba(16, 185, 129, 0.15); color: var(--success); border: 1px solid var(--success); box-shadow: 0 0 20px var(--success-glow); }
  .badge-opacity { background: rgba(244, 63, 94, 0.15); color: var(--danger); border: 1px solid var(--danger); box-shadow: 0 0 20px var(--danger-glow); }
  .badge-warning { background: rgba(245, 158, 11, 0.15); color: var(--warning); border: 1px solid var(--warning); }

  .meter-label { display: flex; justify-content: space-between; font-size: 13px; color: var(--muted); margin-bottom: 6px; }
  .meter-track { height: 12px; background: rgba(255,255,255,0.06); border-radius: 6px; overflow: hidden; position: relative; margin-bottom: 18px; }
  .meter-bar { height: 100%; width: 0%; border-radius: 6px; transition: width 0.8s cubic-bezier(0.16, 1, 0.3, 1), background-color 0.4s ease; }

  .specs-table {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
    background: rgba(0, 0, 0, 0.25);
    border: 1px solid rgba(255,255,255,0.05);
    padding: 14px;
    border-radius: 10px;
    margin-bottom: 18px;
  }
  .spec-item { display: flex; flex-direction: column; gap: 4px; }
  .spec-name { font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.5px; }
  .spec-value { font-size: 15px; font-weight: 700; color: #fff; }

  .clinical-notice {
    border-radius: 10px;
    padding: 14px 16px;
    font-size: 13px;
    line-height: 1.5;
    position: relative;
    border-left: 4px solid;
  }
  .notice-success { background: rgba(16, 185, 129, 0.08); border-color: var(--success); color: #d1fae5; }
  .notice-danger { background: rgba(244, 63, 94, 0.08); border-color: var(--danger); color: #ffe4e6; }
  .notice-warning { background: rgba(245, 158, 11, 0.08); border-color: var(--warning); color: #fef3c7; }

  .footer-stats {
    margin-top: 30px;
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
  }
  @media(max-width: 768px) { .footer-stats { grid-template-columns: repeat(2, 1fr); } }
  .stat-card {
    background: var(--card);
    border: 1px solid var(--border);
    backdrop-filter: blur(12px);
    border-radius: 12px;
    padding: 16px;
    text-align: center;
    transition: transform 0.2s;
  }
  .stat-card:hover { transform: translateY(-3px); }
  .stat-card .label { font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.5px; }
  .stat-card .val { font-size: 20px; font-weight: 800; color: var(--accent); margin-top: 4px; }
</style>
</head>
<body>
<div class="container">
  <header>
    <div class="pulse-indicator">
      <div class="dot"></div> CANLI KLİNİK KARAR DESTEK MOTORU
    </div>
    <h1>Akciğer Röntgeni Teşhis Kokpiti</h1>
    <p>Konumsal Piramit MLP Komitesi, Negatif Log-Olabilirlik Sıcaklık Kalibrasyonu ve Güven Analitiği</p>
  </header>
  
  <div class="grid">
    <!-- SOL: GÖRÜNTÜLEME VE YÜKLEME -->
    <div class="glass-panel">
      <div class="panel-header">
        <h2>DİJİTAL RADYOGRAM GİRİŞİ</h2>
        <span class="badge-icon">DICOM / PNG</span>
      </div>
      
      <div class="upload-box" onclick="document.getElementById('fileInput').click()">
        <svg style="width:36px;height:36px;margin-bottom:8px;fill:var(--accent);opacity:0.8" viewBox="0 0 24 24">
          <path d="M19.35 10.04C18.67 6.59 15.64 4 12 4 9.11 4 6.6 5.64 5.35 8.04 2.34 8.36 0 10.91 0 14c0 3.31 2.69 6 6 6h13c2.76 0 5-2.24 5-5 0-2.64-2.05-4.78-4.65-4.96zM14 13v4h-4v-4H7l5-5 5 5h-3z"/>
        </svg>
        <p style="font-size: 14px; color: var(--text); font-weight: 600;">Röntgen Seçmek veya Sürüklemek İçin Tıklayın</p>
        <span style="font-size: 12px; color: var(--muted); margin-top: 4px; display: block;">veri/val/ dizininden bir göğüs filmi seçebilirsiniz</span>
      </div>
      <input type="file" id="fileInput" accept="image/png, image/jpeg" onchange="handleFile(this)">
      
      <div class="viewer-container" id="viewerContainer">
        <div class="scanner-line" id="scanner"></div>
        <img id="preview" alt="Akciğer Radyogramı">
      </div>
    </div>

    <!-- SAĞ: KARAR DESTEK KOKPİTİ -->
    <div class="glass-panel">
      <div class="panel-header">
        <h2>RADYOLOJİK TANI & GÜVEN PANELİ</h2>
        <span class="badge-icon">INFERENCE ENGINE</span>
      </div>

      <div class="status-placeholder" id="placeholder">
        <svg style="width:48px;height:48px;opacity:0.25;fill:var(--muted)" viewBox="0 0 24 24">
          <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm0 16H5V5h14v14zM8 11h8v2H8zm0-4h8v2H8zm0 8h5v2H8z"/>
        </svg>
        <p>Analiz edilecek bir röntgen görüntüsü bekleniyor...</p>
      </div>

      <div class="result-wrapper" id="resultWrapper">
        <div id="diagnosisBadge"></div>

        <div class="meter-label">
          <span>Kalibre Edilmiş Opasite Olasılığı:</span>
          <span style="font-weight:700;color:var(--text)" id="probText">%0.0</span>
        </div>
        <div class="meter-track">
          <div class="meter-bar" id="meterBar"></div>
        </div>

        <div class="specs-table">
          <div class="spec-item">
            <span class="spec-name">Ham Sinir Ağı Skoru</span>
            <span class="spec-value" id="rawText">%0.0</span>
          </div>
          <div class="spec-item">
            <span class="spec-name">CPU Çıkarım Gecikmesi</span>
            <span class="spec-value" id="latencyText">0.0 ms</span>
          </div>
          <div class="spec-item">
            <span class="spec-name">Klinik Karar Durumu</span>
            <span class="spec-value" id="confidenceStatus" style="color:var(--success)">Stabil</span>
          </div>
          <div class="spec-item">
            <span class="spec-name">Kalibrasyon Katsayısı</span>
            <span class="spec-value" id="tempText">T = 1.83</span>
          </div>
        </div>

        <div id="clinicalNotice" class="clinical-notice"></div>
      </div>
    </div>
  </div>

  <!-- ALT BİLGİ KARTLARI -->
  <div class="footer-stats">
    <div class="stat-card">
      <div class="label">Şampiyon Model F1</div>
      <div class="val">0.9059</div>
    </div>
    <div class="stat-card">
      <div class="label">ECE Kalibrasyon Hatası</div>
      <div class="val">%1.52</div>
    </div>
    <div class="stat-card">
      <div class="label">ECE İyileşme Oranı</div>
      <div class="val">-%71.9</div>
    </div>
    <div class="stat-card">
      <div class="label">Gri Bölge Oranı</div>
      <div class="val">%5.20</div>
    </div>
  </div>
</div>

<script>
function handleFile(input) {
  if (!input.files || !input.files[0]) return;
  const file = input.files[0];

  const reader = new FileReader();
  reader.onload = function(e) {
    const b64Data = e.target.result;
    
    // Görüntü Önizleme ve Lazer Taraması Başlatma
    const img = document.getElementById('preview');
    const container = document.getElementById('viewerContainer');
    const scanner = document.getElementById('scanner');
    img.src = b64Data;
    container.style.display = 'block';
    scanner.style.display = 'block';

    const placeholder = document.getElementById('placeholder');
    const resultWrapper = document.getElementById('resultWrapper');
    placeholder.innerHTML = '<div class="dot" style="margin:0 auto"></div><p style="color:var(--accent);margin-top:10px">Radyogram taranıyor ve çıkarım yapılıyor...</p>';
    placeholder.style.display = 'flex';
    resultWrapper.style.display = 'none';

    fetch('/predict_api', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image_base64: b64Data })
    })
    .then(r => r.json())
    .then(data => {
      scanner.style.display = 'none'; // Tarama tamamlandı
      if (data.error) throw new Error(data.error);

      placeholder.style.display = 'none';
      resultWrapper.style.display = 'block';

      const pCal = data.calibrated_prob_opacity;
      const pRaw = data.raw_prob_opacity;
      const isUncertain = data.is_uncertain;
      const isOpacity = pCal >= 0.50;

      document.getElementById('probText').innerText = '%' + (pCal * 100).toFixed(1);
      document.getElementById('rawText').innerText = '%' + (pRaw * 100).toFixed(1);
      document.getElementById('latencyText').innerText = data.latency_ms.toFixed(1) + ' ms';

      const bar = document.getElementById('meterBar');
      const badge = document.getElementById('diagnosisBadge');
      const notice = document.getElementById('clinicalNotice');
      const confStatus = document.getElementById('confidenceStatus');

      bar.style.width = (pCal * 100) + '%';

      if (isUncertain) {
        badge.className = 'diagnosis-badge badge-warning';
        badge.innerHTML = '⚠️ BELİRSİZ VAKA (GRİ BÖLGE)';
        bar.style.backgroundColor = 'var(--warning)';
        bar.style.boxShadow = '0 0 15px rgba(245, 158, 11, 0.4)';
        confStatus.innerText = 'Düşük Güven';
        confStatus.style.color = 'var(--warning)';
        notice.className = 'clinical-notice notice-warning';
        notice.innerHTML = '<strong>Hekim Karar Destek Uyarısı:</strong> Model güven skoru sınır bölgede (%40 - %60). Tıbbi yapay zekâ etiği ve hasta güvenliği gereği otomatik teşhis üretilmemiştir; uzman radyolog incelemesi önerilir.';
      } else if (isOpacity) {
        badge.className = 'diagnosis-badge badge-opacity';
        badge.innerHTML = '☣️ ŞÜPHELİ OPASİTE (ZATÜRRE)';
        bar.style.backgroundColor = 'var(--danger)';
        bar.style.boxShadow = '0 0 15px var(--danger-glow)';
        confStatus.innerText = 'Yüksek Güven';
        confStatus.style.color = 'var(--danger)';
        notice.className = 'clinical-notice notice-danger';
        notice.innerHTML = '<strong>Yüksek Güvenli Teşhis:</strong> Akciğer sahasında belirgin doku opasitesi ve infiltrasyon paterni saptanmıştır. İleri klinik tetkik tavsiye edilir.';
      } else {
        badge.className = 'diagnosis-badge badge-normal';
        badge.innerHTML = '✅ NORMAL ANATOMİK BULGU';
        bar.style.backgroundColor = 'var(--success)';
        bar.style.boxShadow = '0 0 15px var(--success-glow)';
        confStatus.innerText = 'Yüksek Güven';
        confStatus.style.color = 'var(--success)';
        notice.className = 'clinical-notice notice-success';
        notice.innerHTML = '<strong>Yüksek Güvenli Teşhis:</strong> Akciğer parankiminde ve kostofrenik sinüslerde patolojik bir opasite odağı saptanmamıştır.';
      }
    })
    .catch(err => {
      scanner.style.display = 'none';
      placeholder.innerHTML = '<p style="color:var(--danger)">Hata: ' + err.message + '</p>';
      placeholder.style.display = 'flex';
    });
  };
  reader.readAsDataURL(file);
}
</script>
</body>
</html>
"""

class SafeDemoHandler(http.server.BaseHTTPRequestHandler):
    model = None
    temperature = 1.8347

    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write(HTML_PAGE.encode('utf-8'))

    def do_POST(self):
        if self.path == '/predict_api':
            content_length = int(self.headers.get('Content-Length', 0))
            raw_body = self.rfile.read(content_length)
            
            try:
                payload = json.loads(raw_body.decode('utf-8'))
                b64_str = payload.get('image_base64', '')
                if ',' in b64_str:
                    b64_str = b64_str.split(',', 1)[1]
                
                img_bytes = base64.b64decode(b64_str)
                
                tmp_path = pathlib.Path('_temp_upload.png')
                with open(tmp_path, 'wb') as f:
                    f.write(img_bytes)

                t0 = time.perf_counter()
                feat_vec = features(tmp_path)
                p = probabilities(self.model, feat_vec[None])[0]
                latency = (time.perf_counter() - t0) * 1000

                if tmp_path.exists():
                    tmp_path.unlink()

                p_raw = float(p[1])
                logit = np.log(np.clip(p_raw, 1e-7, 1.0 - 1e-7) / np.clip(1.0 - p_raw, 1e-7, 1.0 - 1e-7))
                p_cal = float(1.0 / (1.0 + np.exp(-logit / self.temperature)))
                is_uncertain = bool(0.40 <= p_cal <= 0.60)

                resp = {
                    "raw_prob_opacity": round(p_raw, 4),
                    "calibrated_prob_opacity": round(p_cal, 4),
                    "is_uncertain": is_uncertain,
                    "latency_ms": round(latency, 2)
                }

                resp_bytes = json.dumps(resp).encode('utf-8')
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(resp_bytes)))
                self.end_headers()
                self.wfile.write(resp_bytes)
            except Exception as e:
                err_text = traceback.format_exc()
                print("Hata detayi:\n", err_text)
                err_resp = json.dumps({"error": str(e)}).encode('utf-8')
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(err_resp)))
                self.end_headers()
                self.wfile.write(err_resp)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='calismalar/deney9/model.npz')
    parser.add_argument('--calib', default='calismalar/calibration_report.json')
    parser.add_argument('--port', type=int, default=8080)
    args = parser.parse_args()

    print(f"Model yukleniyor: {args.model} ...")
    model = load_model(args.model)
    SafeDemoHandler.model = model

    if pathlib.Path(args.calib).exists():
        with open(args.calib, 'r', encoding='utf-8') as f:
            cdata = json.load(f)
            SafeDemoHandler.temperature = float(cdata.get('optimal_temperature', 1.8347))
            print(f"Kalibrasyon Parametresi: T = {SafeDemoHandler.temperature:.4f}")

    print(f"\n========================================================")
    print(f" Ultra-Luks Kokpit Arayuzu Hazir!")
    print(f" Adres: http://localhost:{args.port}")
    print(f"========================================================\n")

    with socketserver.TCPServer(("", args.port), SafeDemoHandler) as httpd:
        httpd.serve_forever()

if __name__ == '__main__':
    main()