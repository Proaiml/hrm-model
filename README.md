# HRM (Hierarchical Reasoning Model) - TinyHRM

Bu depo, insan beynindeki hiyerarşik düşünme (yavaş/stratejik ve hızlı/detay odaklı) mekanizmasını taklit eden **Hierarchical Reasoning Model (HRM)** mimarisinin PyTorch uygulamasını içermektedir.

## 🧠 Model Mimarisi

Model, iki seviyeli (Dual-Level) döngüsel (recurrent) akıl yürütme mimarisi üzerine kurulmuştur:

1. **SwiGLU Katmanı:**
   Modern LLM'lerde (LLaMA vb.) kullanılan kapılı aktivasyon fonksiyonu (`SiLU(W1(x)) * W2(x)`).
2. **HRMBlock:**
   Multi-Head Self-Attention, RMSNorm ve SwiGLU MLP katmanlarını rezidüel bağlantılarla birleştiren temel Transformer bloğu.
3. **ReasoningModule:**
   Birden fazla `HRMBlock` katmanını sıralı çalıştıran ve durum bilgisine girdi enjeksiyonunu (`hidden_state + input_injection`) uygulayan modül.
4. **TinyHrm (Dual-Level Reasoning):**
   * **Düşük Seviye (`L_level`):** `L_cycles` boyunca yerel ve ince adımları çözer.
   * **Yüksek Seviye (`H_level`):** `H_cycles` boyunca düşük seviyenin sentezlediği durumu alarak genel hedef ve makro stratejiyi günceller.

```python
for _ in range(self.H_cycles):
    for _ in range(self.L_cycles):
        z_L = self.L_level(z_L, z_H + x)
    z_H = self.H_level(z_H, z_L)
```

## 🎮 Örnek Görev (8x8 Izgara Yön Tahmini)

Modelin testi için 8x8'lik ızgarada `A` (başlangıç) noktasından `B` (hedef) noktasına ulaşmak için atılması gereken ilk adım yönünü (Yukarı, Aşağı, Sol, Sağ) tahmin eden bir oyuncak problem kurgulanmıştır.

* **A:** 2 (Başlangıç)
* **B:** 3 (Hedef)
* **Hücre Boyutu:** 128 embedding boyutu
* **Yön Tahmini:** `0: YUKARI, 1: AŞAĞI, 2: SOL, 3: SAĞ`

## 🚀 Çalıştırma

Gereksinimler:
```bash
pip install torch matplotlib
```

Modeli çalıştırmak ve eğitim grafiğini görmek için:
```bash
python hrm_model.py
```
