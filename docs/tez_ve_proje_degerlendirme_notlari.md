# Tez ve Proje Değerlendirme Notları

Tarih: 1 Haziran 2026

Bu dosya, proje klasöründeki güncel model çalışmaları, Ubuntu zipinden gelen eski çalışmalar ve mevcut tez PDF/DOCX dosyası incelendikten sonra çıkarılan teknik yorumları içerir.

## Genel Değerlendirme

Bu çalışma lisans tezi için yeterli ve kapsamlıdır. Hatta model sayısı, veri seti çeşitliliği ve deney kapsamı birçok lisans tezinden daha geniştir. Ancak tezin güçlü durması için anlatının doğru kurulması gerekir.

Proje tek bir modelden ziyade iki aşamalı bir sistem olarak anlatılmalıdır:

```text
Görüntü
  -> Detection modeli: panel var mı, varsa nerede?
  -> Panel crop
  -> Classification modeli: panelin durumu nedir?
```

Güncel çalışmada en güçlü doğruluk hattı:

```text
RF-DETR Small detection + FocalNet classification
```

Daha pratik ve hafif kullanım hattı:

```text
YOLOv8s detection + EdgeNeXt classification
```

Ubuntu tarafındaki eski/ek çalışma ise şu şekilde özetlenebilir:

```text
YOLOv11n detection + EfficientNet-B3 / ensemble classification
```

## Tez Hakkında Ana Yorum

Tez dosyası biçim olarak dolu ve kapsamlı görünüyor. Donanım mimarisi, literatür, İHA sistemi, Pixhawk, ESP32-CAM ve saha testi anlatımı tez için değerli. Fakat tezdeki model/veri seti anlatımı güncel klasördeki son deneylerle birebir aynı değil.

En kritik uyumsuzluklar:

1. Tez 6 sınıf anlatıyor:

```text
temiz
tozlu
karlı
kuş pislikli
elektrik hasarlı
fiziksel hasarlı
```

Güncel ana classification çalışması ise 5 sınıftır:

```text
clean
dust
bird_drop
snow
crack_or_damage
```

Burada `electric damage` ayrı sınıf değildir. `crack_or_damage`, kırık/çatlak/hasarlı panel sınıfını temsil eder.

2. Tezde model anlatısı YOLOv8 ile 6 sınıflı sınıflandırma gibi kurulmuş.

Güncel projede YOLOv8s daha çok detection baseline olarak kullanılmıştır. Classification tarafında en iyi model FocalNet'tir.

3. Tezde yaklaşık `%95 Top-1 accuracy` ve `%99.9 Top-5 accuracy` denmiş.

Güncel classification sonucu daha güçlüdür:

| Model | Test Accuracy | Macro F1 |
|---|---:|---:|
| FocalNet | 0.9929 | 0.9937 |
| EdgeNeXt | 0.9905 | 0.9906 |
| MambaVision | 0.9905 | 0.9902 |

Detection tarafında en iyi güncel sonuç:

| Model | mAP50-95 | mAP50 | Precision@IoU50 | Recall@IoU50 | F1@IoU50 |
|---|---:|---:|---:|---:|---:|
| RF-DETR Small | 0.7524 | 0.9140 | 0.8862 | 0.8916 | 0.8889 |

4. Tezde veri seti sayıları farklı yerlerde farklı görünüyor.

Tezde 1500, 3247, 3643, 4005 gibi sayılar geçiyor. Güncel klasörde doğrulanmış ana sayılar şunlardır:

| Görev | Toplam | Train | Val | Test |
|---|---:|---:|---:|---:|
| Classification | 2806 | 1964 | 421 | 421 |
| Detection | 621 | 434 | 93 | 94 |

Ubuntu eski detection çalışmasında ayrıca yaklaşık 3704 görüntülük detection veri seti vardır.

5. Tezde termal kamera bazı gereksinimlerde geçiyor ama kapsam bölümünde termal görüntüleme kapsam dışı denmiş.

Bu çelişki düzeltilmelidir. Güncel proje RGB görüntüler üzerine kuruludur.

## Termal Görüntü Durumu

Ana/güncel veri setinde termal görüntü yoktur. Güncel `gunes_paneli_model` tarafı RGB/normal görüntülerle çalışmaktadır.

Ubuntu zip tarafında eski detection README notunda bir kaynak için `Drone/IR` ifadesi geçmektedir:

```text
solar-panels-detection-sxmhb
```

Ancak işlenmiş klasörde ayrı bir `thermal`, `IR`, `infrared` veya `termal` klasörü yoktur. Bu nedenle tezde net şekilde `termal veri kullandık` demek doğru değildir.

Tez için önerilen ifade:

> Bu çalışmada ana model eğitimleri RGB/normal görüntüler üzerinde gerçekleştirilmiştir. Termal kamera veya termal görüntüleme ana veri setine dahil edilmemiştir.

## Veri Setleri

### Güncel Classification Veri Seti

Kaynaklar:

1. Kaggle `pythonafroz/solar-panel-images`
2. Kaggle `alicjalena/pv-panel-defect-dataset`
3. Roboflow `solar-7u3z6 / solar-faults-detection / v2`
4. Roboflow `faultdetection-j9hnw / solar-panel-pjsbe / v3`
5. Roboflow `solar-panel-4isfg / custom-workflow-multi-label-classification-xy0sf / v1`

İşlenmiş klasör:

```text
gunes_paneli_model/datasets/processed/classification_5class
```

Ağaç ve görüntü sayıları:

```text
classification_5class  (2806)
├── train  (1964)
│   ├── bird_drop  (355)
│   ├── clean  (461)
│   ├── crack_or_damage  (399)
│   ├── dust  (447)
│   └── snow  (302)
├── val  (421)
│   ├── bird_drop  (76)
│   ├── clean  (98)
│   ├── crack_or_damage  (86)
│   ├── dust  (96)
│   └── snow  (65)
└── test  (421)
    ├── bird_drop  (76)
    ├── clean  (99)
    ├── crack_or_damage  (85)
    ├── dust  (96)
    └── snow  (65)
```

### Güncel Detection Veri Seti

Kaynak:

```text
Kaggle fxmikf/solar-panel-bounding-boxes-621
```

İşlenmiş klasör:

```text
gunes_paneli_model/datasets/processed/detection_solar_panel_yolo
```

Ağaç ve görüntü sayıları:

```text
detection_solar_panel_yolo  (621)
└── images  (621)
    ├── train  (434)
    ├── val  (93)
    └── test  (94)
```

Detection sınıfı:

```text
solar_panel
```

## Train / Val / Test Neden Ayrılır?

Veri seti `train`, `val`, `test` olarak ayrılır çünkü modelin gerçekten öğrenip öğrenmediğini adil ölçmek gerekir.

| Bölüm | Kullanım amacı |
|---|---|
| Train | Model bu görüntülerden öğrenir. Ağırlıklar bu veriyle güncellenir. |
| Val | Eğitim sırasında modelin gidişatı kontrol edilir. Hyperparameter tuning, en iyi epoch ve early stopping kararları burada yapılır. |
| Test | En sona saklanır. Model seçildikten sonra nihai ve tarafsız performans ölçümü için kullanılır. |

Basit benzetme:

```text
Train: Model ders çalışıyor.
Val: Deneme sınavı çözülüyor.
Test: Final sınavı.
```

Hepsi train yapılırsa model veriyi ezberleyebilir. Aynı görüntülerle hem eğitim hem ölçüm yapılırsa sonuç yüksek görünür ama gerçek dünyada model kötü çalışabilir. Buna overfitting denir.

Teze yazılabilecek açıklama:

> Veri seti eğitim, doğrulama ve test olmak üzere üç alt kümeye ayrılmıştır. Eğitim seti model parametrelerinin öğrenilmesi için, doğrulama seti hyperparameter seçimi ve eğitim sürecinin izlenmesi için, test seti ise eğitilmiş modelin daha önce görmediği veriler üzerindeki nihai performansını tarafsız şekilde ölçmek için kullanılmıştır. Bu ayrım, modelin ezberleme eğilimini azaltmak ve genelleme başarısını güvenilir biçimde değerlendirmek amacıyla yapılmıştır.

## Word Alan Kodları

DOCX incelenirken `TOC`, `PAGEREF`, `SEQ Tablo` gibi ifadeler göründü. Bunlar hata mesajı değildir; Word'ün otomatik alan kodlarıdır.

| Kod | Anlamı |
|---|---|
| `TOC` | Table of Contents, yani otomatik içindekiler tablosu |
| `PAGEREF` | Bir başlık/şekil/tablonun sayfa numarasına otomatik referans |
| `SEQ Tablo` | Tablo numaralarını otomatik artıran alan |

Normalde belge içinde şöyle görünmelidir:

```text
1. Giriş ................................ 1
Tablo 15 Model Genel Performans Metrikleri
```

Ama alanlar güncellenmezse ham kod gibi görünebilir:

```text
PAGEREF _Toc219898775 \h
SEQ Tablo \* ARABIC
```

Düzeltmek için:

1. DOCX dosyasını Word ile aç.
2. `Ctrl + A` ile tüm belgeyi seç.
3. `F9` tuşuna bas.
4. İçindekiler için sorarsa `Tüm tabloyu güncelle` seç.
5. Eğer kodlar hala görünüyorsa `Alt + F9` yap, sonra tekrar `Ctrl + A` ve `F9`.
6. PDF'i yeniden dışa aktar.

## Tez İçin Önerilen Ana Anlatı

Tezi en savunulabilir hale getirmek için model kısmı şöyle kurulmalıdır:

> Bu çalışmada güneş paneli analizi iki aşamalı olarak ele alınmıştır. İlk aşamada RGB görüntüler üzerinde güneş panellerinin konumu object detection modelleriyle tespit edilmiştir. İkinci aşamada tespit edilen panel bölgeleri classification modelleriyle `clean`, `dust`, `bird_drop`, `snow` ve `crack_or_damage` sınıflarından birine atanmıştır. Detection tarafında RF-DETR Small en yüksek mAP50-95 değerini verirken, classification tarafında FocalNet en yüksek macro F1 değerine ulaşmıştır. Daha hafif ve pratik kullanım için YOLOv8s + EdgeNeXt hattı alternatif olarak değerlendirilmiştir.

Bu anlatı güncel klasördeki gerçek deneylerle uyumludur.

## Sonuç

Çalışma lisans tezi için yeterlidir. Güçlü tarafları:

- İHA/donanım tarafı var.
- Veri toplama ve açık kaynak veri seti kullanımı var.
- Classification ve detection ayrı ele alınmış.
- Birden fazla modern model denenmiş.
- Hyperparameter tuning yapılmış.
- Macro F1, mAP50, mAP50-95, precision, recall ve F1 gibi doğru metrikler kullanılmış.

Dikkat edilmesi gerekenler:

- Tezdeki 6 sınıf anlatısı güncel 5 sınıf çalışmayla uyumlu hale getirilmeli.
- YOLOv8 tek classification modeli gibi anlatılmamalı.
- Termal görüntü kullanıldığı iddia edilmemeli.
- Veri seti sayıları netleştirilmeli.
- Güncel sonuçlar FocalNet, EdgeNeXt, RF-DETR Small ve YOLOv8s üzerinden yazılmalı.
