# Tez Güncelleme Yazım Planı

Tarih: 2 Haziran 2026

Bu dosya, mevcut tez metninin güncel proje çıktılarıyla uyumlu hale getirilmesi için yazım planıdır. Amaç, eski anlatıdaki `YOLOv8 ile 6 sınıflı sınıflandırma` merkezli yapıyı, güncel çalışmaya uygun olan `detection + classification` iki aşamalı sisteme dönüştürmektir.

İstenirse aşağıdaki başlıkların her biri yeniden yazılabilir. Metinler mevcut tez uzunluğuna yakın, akademik üslupta ve doğrudan Word belgesine aktarılabilecek şekilde hazırlanabilir.

## Ana Teknik Eksen

Tezin yeni ana anlatısı şu olmalıdır:

```text
RGB/drone görüntüsü
    -> Detection modeli ile güneş paneli tespiti
    -> Panel bounding box / crop
    -> Classification modeli ile panel durum sınıflandırması
    -> clean / dust / bird_drop / snow / crack_or_damage
```

Güncel önerilen model hattı:

| Kullanım amacı | Detection | Classification |
|---|---|---|
| En yüksek doğruluk | RF-DETR Small | FocalNet |
| Daha hafif/pratik kullanım | YOLOv8s | EdgeNeXt |

Ana veri seti ve sonuçlar:

| Görev | Veri | En iyi model | Ana sonuç |
|---|---:|---|---:|
| Classification | 2806 görüntü | FocalNet | Macro F1 = 0.9937 |
| Detection | 621 görüntü | RF-DETR Small | mAP50-95 = 0.7524 |

## Yazılabilir Bölümler ve Öncelik

### 1. ÖZET

Öncelik: Çok yüksek

Mevcut sorun:

- 6 sınıf anlatıyor.
- Elektrik hasarı ayrı sınıf gibi geçiyor.
- YOLOv8 merkezli eski anlatıya yakın.
- Güncel iki aşamalı pipeline yeterince görünmüyor.

Yeni içerik:

- RGB/İHA görüntüleri ile güneş paneli analizi.
- Detection ve classification iki aşamalı yapı.
- Detection tarafında panel konumunun bulunması.
- Classification tarafında 5 sınıf:

```text
clean
dust
bird_drop
snow
crack_or_damage
```

- En iyi modeller:

```text
RF-DETR Small
FocalNet
```

- Hafif alternatif:

```text
YOLOv8s + EdgeNeXt
```

Yaklaşık uzunluk:

- Mevcut özet uzunluğuna yakın, 1 uzun paragraf.
- 250-350 kelime arası.

Yazım notu:

- Termal görüntü kullanıldı denmemeli.
- Elektriksel hot-spot doğrudan tespit edildi denmemeli.
- `RGB görüntüler` vurgusu yapılmalı.

### 2. ABSTRACT

Öncelik: Çok yüksek

Mevcut sorun:

- Türkçe özetle aynı eski teknik anlatıyı İngilizce tekrar ediyor.

Yeni içerik:

- ÖZET bölümünün İngilizce karşılığı.
- `two-stage deep learning pipeline`, `object detection`, `condition classification`, `RGB UAV images` ifadeleri kullanılmalı.

Yaklaşık uzunluk:

- 250-350 kelime.

Yazım notu:

- Teknik terimler doğru kullanılmalı:

```text
object detection
classification
bounding box
macro F1
mAP50-95
```

### 3. 1.2.5 Projenin Temel Amacı

Öncelik: Çok yüksek

Mevcut sorun:

- Projenin amacı 6 sınıflı tek model gibi duruyor.

Yeni içerik:

- Amaç iki parçaya ayrılmalı:

```text
1. Güneş panelinin görüntüdeki konumunu tespit etmek
2. Tespit edilen panelin durumunu sınıflandırmak
```

- RGB görüntü kısıtı açıkça yazılmalı.
- Bakım önceliklendirmesine katkı anlatılmalı.

Yaklaşık uzunluk:

- 2-4 paragraf.
- 250-400 kelime.

### 4. 1.2.6 Alt Hedefler

Öncelik: Yüksek

Mevcut sorun:

- Hedefler eski YOLOv8/6 sınıf yapısına bağlı kalmış olabilir.

Yeni alt hedefler:

1. RGB tabanlı classification veri setinin hazırlanması.
2. Panel detection veri setinin hazırlanması.
3. Classification için farklı modern mimarilerin eğitilmesi.
4. Detection için farklı model ailelerinin eğitilmesi.
5. Hyperparameter tuning ile uygun parametrelerin seçilmesi.
6. Modellerin ortak metriklerle karşılaştırılması.
7. En iyi doğruluk ve pratik kullanım alternatiflerinin belirlenmesi.
8. Detection çıktısından classification aşamasına geçecek pipeline tasarımının ortaya konması.

Yaklaşık uzunluk:

- Madde madde veya kısa açıklamalı liste.
- 300-500 kelime.

### 5. 1.2.7 Proje Kapsamı

Öncelik: Çok yüksek

Mevcut sorun:

- Termal kamera/görüntüleme ile RGB kapsamı arasında çelişki var.

Yeni içerik:

- Kapsamda olanlar:

```text
RGB görüntü analizi
Güneş paneli detection
Panel durum classification
Açık kaynak veri setlerinin birleştirilmesi
Hyperparameter tuning
Model karşılaştırması
```

- Kapsam dışında olanlar:

```text
Termal görüntüleme
Multispektral/hiperspektral analiz
Elektriksel ölçüm
Panel temizleme robotu
Hot-spot tespiti
```

Yaklaşık uzunluk:

- Mevcut kapsam bölümüne yakın.
- 500-800 kelime.
- Tablo da eklenebilir.

### 6. 1.3 Projenin Kısıtları ve Varsayımları

Öncelik: Orta-yüksek

Yeni eklenecek kısıtlar:

- RGB görüntüler ışık, gölge, açı ve hava koşullarından etkilenebilir.
- Termal görüntü olmadığı için elektriksel hot-spot arızaları doğrudan tespit edilmez.
- Veri setleri farklı kaynaklardan geldiği için domain farkı olabilir.
- Detection veri seti classification veri setinden ayrıdır.
- Bir görüntüde panel yoksa classification aşaması çalıştırılmamalıdır.

Yaklaşık uzunluk:

- 400-700 kelime.

### 7. 1.3.1 Başarı Kriterleri

Öncelik: Yüksek

Mevcut sorun:

- Tek bir doğruluk/mAP hedefi gibi yazılmış olabilir.

Yeni başarı kriterleri:

Classification için:

```text
Accuracy
Macro F1
Macro Precision
Macro Recall
```

Detection için:

```text
mAP50
mAP50-95
Precision@IoU50
Recall@IoU50
F1@IoU50
```

Yaklaşık uzunluk:

- Kısa açıklama + tablo.
- 250-400 kelime.

### 8. 3.2 Veri Hazırlama

Öncelik: Çok yüksek

Mevcut sorun:

- Veri sayıları karışık.
- 6 sınıf ve eski veri seti anlatısı baskın.

Yeni yapı:

```text
3.2.1 Classification Veri Seti
3.2.2 Detection Veri Seti
3.2.3 Veri Temizleme ve Standardizasyon
3.2.4 Train / Validation / Test Ayrımı
```

Classification veri seti:

```text
Toplam: 2806 görüntü
Train: 1964
Validation: 421
Test: 421
```

Detection veri seti:

```text
Toplam: 621 görüntü
Train: 434
Validation: 93
Test: 94
```

Yaklaşık uzunluk:

- 900-1400 kelime.
- 2 tablo eklenmeli.

### 9. 3.2.4 Train / Validation / Test Ayrımı

Öncelik: Yüksek

Eklenecek açıklama:

> Veri seti eğitim, doğrulama ve test olmak üzere üç alt kümeye ayrılmıştır. Eğitim seti model parametrelerinin öğrenilmesi için, doğrulama seti hyperparameter seçimi ve eğitim sürecinin izlenmesi için, test seti ise eğitilmiş modelin daha önce görmediği veriler üzerindeki nihai performansını tarafsız şekilde ölçmek için kullanılmıştır. Bu ayrım, modelin ezberleme eğilimini azaltmak ve genelleme başarısını güvenilir biçimde değerlendirmek amacıyla yapılmıştır.

Yaklaşık uzunluk:

- 1-2 paragraf.
- 150-250 kelime.

### 10. 3.3 Modelin Geliştirilmesi

Öncelik: Çok yüksek

Mevcut sorun:

- YOLOv8 tek ana model gibi anlatılıyor.

Yeni yapı:

```text
3.3.1 Genel Pipeline Tasarımı
3.3.2 Detection Modelleri
3.3.3 Classification Modelleri
3.3.4 Hyperparameter Tuning
3.3.5 Model Seçim Kriterleri
```

Detection modelleri:

```text
YOLOv8s
YOLOv11s
YOLOv12s
RT-DETR-L
RF-DETR Small
YOLO-NAS-S
EfficientDet-D1
EfficientDet-D2
Deformable DETR
Grounding-DINO Tiny
```

Classification modelleri:

```text
FocalNet
EdgeNeXt
MambaVision
FastViT
RepViT
TinyViT
MaxViT
PoolFormer
CoAtNet
SigLIP
CLIP
HorNet
```

Yaklaşık uzunluk:

- 1200-1800 kelime.
- Model tabloları eklenebilir.

### 11. 3.3.4 Hyperparameter Tuning

Öncelik: Çok yüksek

Classification tuning:

| Trial | Learning rate | Batch size | Weight decay | Tuning epoch |
|---:|---:|---:|---:|---:|
| 1 | 0.0001 | 8 | 0.0001 | 3 |
| 2 | 0.0003 | 8 | 0.0001 | 3 |
| 3 | 0.0001 | 16 | 0.0005 | 3 |

Detection YOLO/RT-DETR tuning:

| Trial | lr0 | Batch | Tuning epoch |
|---:|---:|---:|---:|
| 1 | 0.01 | 16 | 5 |
| 2 | 0.005 | 16 | 5 |
| 3 | 0.01 | 8 | 5 |

RF-DETR tuning:

| Trial | LR | Batch | Grad Accum | Epoch |
|---:|---:|---:|---:|---:|
| 1 | 0.0001 | 1 | 16 | 5 |
| 2 | 0.00005 | 1 | 16 | 5 |
| 3 | 0.0001 | 2 | 8 | 5 |

Yaklaşık uzunluk:

- 500-900 kelime.
- Tablo ağırlıklı olmalı.

### 12. 4.1 Gereksinimler

Öncelik: Orta

Değişmesi gereken yer:

Eski:

```text
Termal kamera ile paneller üzerindeki sıcaklık anomalileri tespit edilmelidir.
```

Yeni:

```text
RGB kamera ile panel yüzeyi görüntüleri alınmalı ve yer istasyonuna aktarılmalıdır.
```

YOLOv8 ile 6 sınıf ifadesi varsa şu hale gelmeli:

```text
Detection modeli ile panel konumu tespit edilmeli, classification modeli ile panel durumu sınıflandırılmalıdır.
```

Yaklaşık uzunluk:

- Tablo satırları düzenlenecek.
- Uzun metin gerekmez.

### 13. 4.3 Model Performans Sonuçları

Öncelik: Çok yüksek

Mevcut sorun:

- YOLOv8 Top-1/Top-5 sonuçları güncel deneyleri temsil etmiyor.

Yeni yapı:

```text
4.3.1 Classification Sonuçları
4.3.2 Detection Sonuçları
4.3.3 Model Seçimi ve Yorum
```

Classification sonuç tablosu:

| Model | Test Accuracy | Macro F1 | Best Val Acc |
|---|---:|---:|---:|
| FocalNet | 0.9929 | 0.9937 | 0.9905 |
| EdgeNeXt | 0.9905 | 0.9906 | 0.9905 |
| MambaVision | 0.9905 | 0.9902 | 0.9881 |
| FastViT | 0.9834 | 0.9831 | 0.9929 |
| RepViT | 0.9786 | 0.9786 | 0.9929 |

Detection sonuç tablosu:

| Model | mAP50-95 | mAP50 | F1@IoU50 |
|---|---:|---:|---:|
| RF-DETR Small | 0.7524 | 0.9140 | 0.8889 |
| Deformable DETR | 0.6232 | 0.8224 | 0.7351 |
| YOLO-NAS-S | 0.6119 | 0.8596 | 0.5792 |
| EfficientDet-D2 | 0.6094 | 0.8395 | 0.8242 |
| YOLOv8s | 0.6093 | 0.8307 | 0.7500 |

Yaklaşık uzunluk:

- 800-1200 kelime.
- Tablolar + açıklayıcı yorum.

### 14. 4.4 Sistem Entegrasyon Sonuçları

Öncelik: Orta

Mevcut metin büyük ölçüde kalabilir. Sadece model yönlendirme cümleleri güncellenmeli.

Eski:

```text
YOLOv8 modeline yönlendirilir.
```

Yeni:

```text
Önce detection modeline, ardından tespit edilen panel bölgesi classification modeline yönlendirilir.
```

Yaklaşık uzunluk:

- 1-2 paragraf düzenleme yeterli.

### 15. 4.5 Gerçek Zamanlı Panel Tespiti Sonuçları

Öncelik: Orta

Yeni içerik:

- OpenCV ön filtreleme anlatısı kalabilir.
- Ancak nihai panel tespiti detection model ailesiyle ilişkilendirilmeli.
- `panel var/yok` ve `bounding box` ayrımı yapılmalı.

Yaklaşık uzunluk:

- 300-500 kelime.

### 16. 4.6 Saha Testi Sonuçları

Öncelik: Orta

Değişmesi gerekenler:

- 6 kategori yerine 5 classification sınıfı.
- Elektrik hasarı doğrudan tespit edildi iddiası çıkarılmalı.
- ESP32-CAM ile toplanan görüntülerin RGB olduğu vurgulanmalı.

Yaklaşık uzunluk:

- 300-600 kelime.

### 17. 4.7 İş Paketleri

Öncelik: Düşük-orta

Değişmesi gerekenler:

- `YOLOv8 modelinin transfer öğrenme ile eğitilmesi ve %95 doğruluk` gibi ifadeler güncel sonuçlarla değişmeli.
- AI iş paketi şu hale getirilmeli:

```text
Detection ve classification veri setlerinin hazırlanması,
farklı model ailelerinin eğitilmesi,
hyperparameter tuning yapılması,
RF-DETR Small + FocalNet ve YOLOv8s + EdgeNeXt hatlarının karşılaştırılması.
```

### 18. 4.8 Beklenen Çıktılar

Öncelik: Orta

Değişmesi gerekenler:

- `6 sınıflı YOLOv8` yerine `iki aşamalı detection + classification sistemi`.
- `elektriksel hasar` yerine `crack_or_damage`.
- Başarı kriterleri classification ve detection için ayrı yazılmalı.

Yaklaşık uzunluk:

- Tablo satırları güncellenecek.

### 19. 5. Tartışma

Öncelik: Çok yüksek

Yeni ana yorum:

> Classification tarafında FocalNet en yüksek macro F1 değerini vermiştir. EdgeNeXt modeli ise FocalNet'e çok yakın sonuç üretmesine rağmen çok daha küçük checkpoint boyutuna sahip olduğu için pratik kullanımda güçlü bir alternatiftir. Detection tarafında RF-DETR Small en yüksek mAP50-95 ve dengeli F1 değerini sağlamıştır. YOLOv8s ise daha kolay deploy edilebilir yapısı nedeniyle pratik baseline olarak değerlendirilmiştir.

Yaklaşık uzunluk:

- 700-1200 kelime.

### 20. 5.1 Sonuçların Değerlendirilmesi

Öncelik: Çok yüksek

Eski YOLOv8 %95 anlatısı çıkarılmalı.

Yeni içerik:

- Classification sonucu:

```text
FocalNet Macro F1 = 0.9937
EdgeNeXt Macro F1 = 0.9906
```

- Detection sonucu:

```text
RF-DETR Small mAP50-95 = 0.7524
RF-DETR Small F1@IoU50 = 0.8889
```

Yaklaşık uzunluk:

- 400-700 kelime.

### 21. 5.2 Literatür ile Karşılaştırma

Öncelik: Yüksek

Tablo 20 güncellenmeli.

Yeni satır:

| Çalışma | Model | Sonuç |
|---|---|---|
| Bu proje | RF-DETR Small + FocalNet | Detection mAP50-95 = 0.7524, Classification Macro F1 = 0.9937 |

Yazım notu:

- Classification accuracy ile detection mAP aynı şey değildir.
- Literatür karşılaştırmasında bu fark açıkça belirtilmeli.

Yaklaşık uzunluk:

- 500-800 kelime.

### 22. 5.4 Sınırlılıklar ve Karşılaşılan Zorluklar

Öncelik: Yüksek

Eklenecek sınırlılıklar:

- Termal görüntü kullanılmamıştır.
- Elektriksel hot-spot arızaları doğrudan tespit edilmemiştir.
- RGB görüntüler ışık, açı, gölge ve hava koşullarına duyarlıdır.
- Detection veri setinde train-test duplicate kontrolü yapılmış, ortak değerlendirmede 1 duplicate test görüntüsü çıkarılmıştır.
- Detection ve classification veri setleri farklı kaynaklardan geldiği için domain farkı olabilir.

Yaklaşık uzunluk:

- 500-800 kelime.

### 23. 5.5 Gelecek Çalışmalar ve İyileştirme Önerileri

Öncelik: Orta-yüksek

Eklenecek öneriler:

- Termal kamera entegrasyonu.
- Hot-spot tespiti.
- Uçtan uca inference pipeline.
- Detection çıktısından otomatik crop alma.
- Daha büyük, dengeli ve gerçek saha verisi içeren veri seti.
- Edge cihazlarda model optimizasyonu.
- Temizlenmiş detection split ile yeniden eğitim.
- LTE/5G haberleşme.

Yaklaşık uzunluk:

- 500-800 kelime.

## Zaman Azsa Önce Yazılacak Başlıklar

Önce şu başlıklar yazılmalı:

```text
ÖZET
ABSTRACT
1.2.5 Projenin Temel Amacı
1.2.7 Proje Kapsamı
3.2 Veri Hazırlama
3.3 Modelin Geliştirilmesi
4.3 Model Performans Sonuçları
5.1 Sonuçların Değerlendirilmesi
5.2 Literatür ile Karşılaştırma
5.4 Sınırlılıklar ve Karşılaşılan Zorluklar
```

Bu başlıklar düzelirse tezin ana teknik omurgası güncel proje ile uyumlu hale gelir.

## Yazım Sırası Önerisi

1. Önce `3.2 Veri Hazırlama` yazılmalı.
2. Sonra `3.3 Modelin Geliştirilmesi` yazılmalı.
3. Ardından `4.3 Model Performans Sonuçları` yazılmalı.
4. Sonra `5.1`, `5.2`, `5.4`, `5.5` tartışma bölümleri yazılmalı.
5. En son `ÖZET` ve `ABSTRACT` güncellenmeli.

Bu sıra daha doğrudur çünkü özet, yöntem ve sonuçlar netleştikten sonra yazılırsa daha tutarlı olur.

## Benden İstenebilecek Yazım Komutları

Aşağıdaki gibi bölüm bölüm istenebilir:

```text
3.2 Veri Hazırlama bölümünü tez dilinde yeniden yaz.
```

```text
3.3 Modelin Geliştirilmesi bölümünü mevcut tez uzunluğuna yakın şekilde yaz.
```

```text
4.3 Model Performans Sonuçları bölümünü tablolarıyla birlikte yaz.
```

```text
ÖZET ve ABSTRACT bölümlerini güncel proje sonuçlarına göre yeniden yaz.
```

```text
5. Tartışma bölümünü FocalNet, EdgeNeXt, RF-DETR Small ve YOLOv8s sonuçlarına göre yeniden yaz.
```

## Not

Uzun bölümler tek seferde yazılabilir; ancak Word'e daha kolay aktarmak için bölüm bölüm yazmak daha sağlıklı olur. Her bölüm mevcut tezin akademik tonuna yakın, benzer uzunlukta ve doğrudan yapıştırılabilir biçimde hazırlanabilir.
