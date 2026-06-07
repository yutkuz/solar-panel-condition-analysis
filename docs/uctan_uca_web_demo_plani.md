# Uctan Uca Gunes Paneli Web Demo Plani

## 1. Platform Karari

Demo **web uygulamasi** olarak gelistirilecektir.

Web tercihinin nedenleri:

- Tez sunumunda ayni bilgisayarda tarayicidan kolayca calistirilabilir.
- Telefon, emulator, APK imzalama ve cihaz baglantisi gibi ek riskler olusturmaz.
- Mevcut Python modelleri dogrudan sunucu tarafinda kullanilabilir.
- Masaustu ve mobil ekranlara uyumlu tek bir arayuz yeterlidir.
- Uygulama API tabanli kurulacagi icin daha sonra mobil uygulama eklenebilir.

Ilk surumde ayri bir mobil uygulama yapilmayacaktir. Web arayuzu responsive olacak ve telefondan da kullanilabilecektir.

## 2. Demonun Amaci

Kullanici bir gunes paneli fotografi yukleyecek. Sistem:

1. Fotograftaki gunes paneli veya panellerini tespit edecek.
2. Her paneli ayri kirpacak.
3. Her kirpilan paneli durum siniflandirma modeline verecek.
4. Sonucu fotograf uzerinde kutu, sinif ve guven skoru ile gosterecek.
5. Sonuclari JSON ve isaretlenmis fotograf olarak indirebilecek.

Siniflandirma siniflari:

- `bird_drop`: Kus pisligi
- `clean`: Temiz
- `crack_or_damage`: Catlak veya hasar
- `dust`: Toz
- `snow`: Kar

## 3. Kullanilacak Modeller

### Dogruluk Profili

- Panel tespiti: `experiments/detection/rfdetr_small/final/gpu/checkpoint_best_total.pth`
- Durum siniflandirma: `experiments/classification/focalnet/final/best.pt`

Bu profil tez sunumunda ana sonuc olarak kullanilacaktir.

### Pratik Profil

- Panel tespiti: `experiments/detection/yolov8s/final/weights/best.pt`
- Durum siniflandirma: `experiments/classification/edgenext/final/best.pt`

Bu profil daha dusuk kaynak tuketimi ve hizli sonuc icin alternatif olacaktir.

Model dosyalari uygulama baslatilirken bir kez bellekte yuklenecektir. CUDA bulunursa GPU, bulunmazsa CPU kullanilacaktir.

## 4. Teknik Yapi

Onerilen teknoloji yigini:

- Backend ve API: FastAPI
- Sunucu: Uvicorn
- Arayuz: HTML, CSS ve vanilla JavaScript
- Sablon: Jinja2
- Siniflandirma: PyTorch ve timm
- YOLO tespiti: ultralytics
- RF-DETR tespiti: rfdetr
- Goruntu islemleri: Pillow ve OpenCV

React gibi ayri bir derleme sistemi ilk surum icin gerekli degildir. Bu yapi kurulum ve sunum riskini azaltir.

Onerilen klasor yapisi:

```text
demo_web/
|-- app.py
|-- inference/
|   |-- classifier.py
|   |-- detector.py
|   |-- pipeline.py
|   `-- schemas.py
|-- templates/
|   `-- index.html
|-- static/
|   |-- app.js
|   `-- styles.css
`-- outputs/

demo_samples/
`-- external/
    |-- bird_drop/
    |-- clean/
    |-- crack_or_damage/
    |-- dust/
    |-- snow/
    `-- manifest.json

scripts/
|-- collect_external_test_images.py
`-- evaluate_external_demo_set.py
```

## 5. Uctan Uca Tahmin Akisi

1. JPG, PNG veya WebP fotograf okunur ve RGB bicimine donusturulur.
2. Secilen tespit modeli fotograf uzerinde calistirilir.
3. Dusuk guvenli tespitler elenir.
4. Her panel kutusuna kucuk bir kenar boslugu eklenerek panel kirpilir.
5. Kirpilan goruntu `224x224` boyutuna getirilir ve ImageNet normalizasyonu uygulanir.
6. Secilen siniflandirma modeli her panel icin calistirilir.
7. Softmax sonucundan sinif, guven skoru ve sinif olasiliklari alinir.
8. Tespit kutulari, Turkce sinif adlari ve guven degerleri fotograf uzerine cizilir.
9. Isaretlenmis fotograf, panel bazli sonuclar ve calisma suresi arayuze dondurulur.

Birden fazla panel varsa her panel ayri siniflandirilacaktir. Panel bulunamazsa sistem bunu acikca bildirecek ve tum fotografi panelmis gibi sessizce siniflandirmayacaktir.

## 6. Arayuz

Ana ekran dogrudan kullanilabilir demo olacaktir; ayri bir tanitim sayfasi yapilmayacaktir.

Arayuzde bulunacak kontroller:

- Surukle-birak ve dosya secme alani
- `Dogruluk` ve `Pratik` model profili secimi
- `Tam Akis` ve `Sadece Siniflandirma` modu
- Tespit guven esigi
- Tahmini baslatma ve sonucu temizleme komutlari
- Orijinal ve isaretlenmis fotograf gorunumu
- Her panel icin kirpim, durum, guven ve sinif olasiliklari
- Toplam, tespit ve siniflandirma calisma sureleri
- Isaretlenmis fotografi ve JSON sonucunu indirme
- Model yukleniyor, isleniyor, hata ve panel bulunamadi durumlari

`Sadece Siniflandirma` modu, panelin yakin plan cekildigi ve tespit modelinin kutu uretmesinin beklenmedigi fotograflar icin kullanilacaktir.

Ilk deger olarak tespit esigi `0.25` kullanilabilir. Arayuzde ayarlanabilir olacaktir. Siniflandirma guveni `0.60` altinda ise sonuc `dusuk guven` olarak isaretlenecek, ancak tahmin gizlenmeyecektir.

## 7. Internetten Harici Test Goruntuleri

Her sinif icin hedef **3-5 fotograf**, toplamda **15-25 harici fotograf** toplanacaktir. Bunlar egitim veya resmi test verisine eklenmeyecek; modelin gercek dunya kosullarina karsi nitel bir stres testi olarak kullanilacaktir.

Oncelikli kaynak Wikimedia Commons gibi lisansi ve kaynak bilgisi acik platformlardir.

Ornek aday kaynaklar:

- Genel ve temiz paneller: <https://commons.wikimedia.org/wiki/Category:Solar_panels>
- Karli paneller: <https://commons.wikimedia.org/wiki/Category:Snow_on_solar_panels>
- Kirik paneller: <https://commons.wikimedia.org/wiki/Category:Broken_solar_panels>
- Tozlu panel: <https://commons.wikimedia.org/wiki/File:Dusty_solar_panel.jpg>
- Catlak panel: <https://commons.wikimedia.org/wiki/File:Cracked_solar_panel.jpg>
- Karli panel: <https://commons.wikimedia.org/wiki/File:Solar_panels_with_snow.jpg>

Kus pisligi sinifi icin acik lisansli ve panel durumunu net gosteren fotograf bulunamazsa:

- Kullanici tarafindan cekilmis ve kullanma izni bulunan fotograflar kullanilacak veya
- Mevcut acik veri kaynagindan, egitim kumesinde yer almayan ornekler ayrilacaktir.

Ticari sitelerdeki gorseller, acik bir lisans veya izin yoksa proje ZIP dosyasina konulmayacaktir.

Her dosya icin `manifest.json` icinde su bilgiler tutulacaktir:

```json
{
  "id": "snow_001",
  "expected_class": "snow",
  "source_page": "https://...",
  "image_url": "https://...",
  "license": "CC BY-SA 4.0",
  "author": "...",
  "downloaded_at": "YYYY-MM-DD",
  "sha1": "...",
  "notes": "Panelin buyuk bolumu karla kapli."
}
```

Indirilen fotograflarin SHA-1 degerleri mevcut egitim, dogrulama ve test fotograflariyla karsilastirilacaktir. Eslesen dosyalar harici testten cikarilacaktir.

## 8. Harici Test Yontemi

Harici fotograflar iki grupta test edilecektir:

- **Tam akis testi:** Panelin tamamini veya buyuk bolumunu gosteren fotograflarda tespit ve siniflandirma birlikte calistirilir.
- **Siniflandirma testi:** Yakin plan veya sadece panel yuzeyini gosteren fotograflar dogrudan siniflandirilir.

Raporlanacak degerler:

- Tespit edilen panel sayisi ve tespit basari orani
- Tespit edilen kirpimlerde siniflandirma dogrulugu
- Uctan uca dogru sonuc orani
- Sinif bazli dogru ve yanlis tahminler
- Tahmin guvenleri
- Fotograf ve panel basina calisma suresi
- Basarisiz ornekler ve muhtemel nedenleri

Bu sonuclar `harici test` olarak adlandirilacak ve mevcut test kumesi metriklerinin yerine gecmeyecektir. Az sayida internet fotografiyla elde edilen oranlar genelleme iddiasi olarak sunulmayacaktir.

## 9. Gelistirme Asamalari

### Asama 1 - Ortam ve Bagimliliklar

- Mevcut checkpoint dosyalarinin yuklenmesini dogrulama
- FastAPI, Uvicorn, Jinja2 ve `python-multipart` bagimliliklarini ekleme
- YOLO ve RF-DETR paketlerinin temiz ortamda kurulumunu dogrulama
- GPU ve CPU calisma yollarini test etme

### Asama 2 - Tahmin Cekirdegi

- Siniflandirma yukleme ve tahmin modulunu ayirma
- YOLO ve RF-DETR icin ortak tespit arayuzu olusturma
- Tespit, kirpma, siniflandirma ve gorsellestirme hattini kurma
- Model yuklemeyi onbellegi kullanacak sekilde yapma

### Asama 3 - API

- Saglik ve model durumu endpoint'i
- Tek fotograf tahmin endpoint'i
- Profil, mod ve esik parametreleri
- Yapilandirilmis JSON ve isaretlenmis fotograf ciktisi
- Dosya tipi, boyut ve hata kontrolleri

### Asama 4 - Web Arayuzu

- Responsive yukleme ve sonuc ekrani
- Model profili ve mod secimi
- Panel bazli sonuc listesi
- Bekleme, hata ve bos sonuc durumlari
- Fotograf ve JSON indirme

### Asama 5 - Harici Test Verileri

- Acik lisansli adaylari secme
- Kaynak ve lisans manifestini olusturma
- Veri sizintisi icin hash kontrolu
- Her fotograf icin beklenen sinifi elle dogrulama

### Asama 6 - Dogrulama

- Her iki model profilini calistirma
- Tek panel, coklu panel ve panelsiz fotograf testleri
- Bes durum sinifinin harici fotograflarla testi
- Hata durumlari ve buyuk dosya testi
- Masaustu ve mobil ekranlarda tarayici testi
- Sonuc ekranlarinin tezde kullanilmak uzere goruntulenmesi

### Asama 7 - Paketleme ve Tez Entegrasyonu

- Calistirma komutlari ve sistem gereksinimleri
- Harici test tablosu ve hata ornekleri
- Mimari ve veri akisi semasi
- Demo ekran goruntuleri
- Model dosyalari haric tutulacaksa indirme/yol aciklamasi

## 10. Kabul Kriterleri

Demo tamamlanmis sayilmak icin:

- JPG, PNG ve WebP yukleyebilmelidir.
- Dogruluk ve Pratik profilleri calismalidir.
- Tam akis ve sadece siniflandirma modlari calismalidir.
- Birden fazla paneli ayri ayri raporlayabilmelidir.
- Panel bulunamadiginda dogru mesaj gostermelidir.
- Sonuclari fotograf ve JSON olarak indirebilmelidir.
- En az 3, hedef 5 olmak uzere her siniftan harici fotograf test edilmelidir.
- Harici fotograflarin kaynak ve lisans bilgileri kaydedilmelidir.
- Harici test fotograflariyla egitim verileri arasinda hash kontrolu yapilmalidir.
- CPU uzerinde de calisabilmeli; yavaslik arayuzde yonetilmelidir.
- Masaustu ve mobil gorunumlarda tasma ve ust uste binme olmamalidir.
- Kurulum ve calistirma adimlari tekrar edilebilir olmalidir.

## 11. Riskler ve Onlemler

- **RF-DETR kurulumu veya CPU hizi:** YOLOv8s tabanli Pratik profil yedek olarak kullanilacak.
- **Internet fotograflarinin belirsiz etiketi:** Her fotograf elle incelenecek ve supheli ornekler metrik disinda tutulacak.
- **Kus pisligi fotografi azligi:** Acik lisansli kaynak, kullanici fotografi veya egitimde kullanilmamis acik veri ornegi aranacak.
- **Panel tespitinin yakin planda basarisiz olmasi:** Sadece Siniflandirma modu sunulacak.
- **Buyuk model dosyalari:** Baslangicta yukleme durumu gosterilecek ve modeller istek basina yeniden yuklenmeyecek.
- **Veri sizintisi:** Dosya hash kontrolu ve kaynak kaydi zorunlu olacak.

## 12. Teslim Edilecek Ciktilar

- Calisan responsive web demo
- Uctan uca tahmin API'si
- Iki model profili
- Kaynak ve lisans manifestli harici test klasoru
- Harici test sonuc raporu
- Kurulum ve calistirma dokumani
- Tez icin mimari sema, sonuc tablosu ve ekran goruntuleri

Mobil uygulama ancak web demo, API ve harici testler tamamlandiktan sonra ek calisma olarak dusunulmelidir. Bitirme projesi acisindan asil katki, ayni modellerin gercek fotograflarda tekrar edilebilir bicimde calistirilmasi ve basari/basarisizliklarinin raporlanmasidir.
