# Karakter ve BPE Tokenizer ile Minyatür Dil Modeli Eğitimi

Bu projede Türkçe metinler üzerinde iki farklı tokenizasyon yöntemi kullanılarak minyatür Transformer modelleri eğitilmiştir.

Çalışma iki aşamadan oluşmaktadır:

1. Başkent isimlerinin karakter karakter tokenlaştırılması
2. Türkçe bir paragraf üzerinde BPE tokenizer eğitilmesi

Her iki aşamada da eğitim amaçlı hazırlanmış TinyQwen ve TinyGemma modelleri kullanılmıştır.

> Bu projedeki modeller gerçek Qwen veya Gemma modelleri değildir. Qwen ve Gemma mimarilerinden esinlenilmiş küçük, eğitim amaçlı Transformer uygulamalarıdır.

---

## Projenin Gelişim Süreci

### 1. Karakter seviyesinde başkent üretme

İlk aşamada dünya başkentlerinden oluşan bir veri kümesi hazırlanmıştır.

Ham veri üzerinde şu işlemler uygulanmıştır:

- Boş ve geçersiz satırlar temizlendi
- İsimler standart biçime getirildi
- Her satırda tek bir başkent adı olacak şekilde veri hazırlandı
- Başkent isimleri karakter karakter tokenlaştırıldı

Örnek eğitim verisi:

```text
ankara
londra
berlin
tokyo
paris
madrid
```

Karakter tokenizer her harfi ayrı bir token olarak ele almaktadır.

Örneğin:

```text
ankara
```

metni yaklaşık olarak şu şekilde parçalanır:

```text
a | n | k | a | r | a
```

Her karakter vocabulary içerisinde bir token ID ile temsil edilir.

Bu aşamada model, bir karakterden sonra hangi karakterin geleceğini tahmin etmeyi öğrenmiştir.

Örneğin:

```text
a → n
n → k
k → a
```

Bu yöntemle TinyQwen ve TinyGemma modelleri başkent isimleri üzerinde eğitilmiş ve eğitim verisindeki isim yapılarına benzeyen yeni isimler üretmeleri denenmiştir.

---

## Karakter Tokenizer Akışı

```text
Başkent listesi
      ↓
Veri temizleme
      ↓
Her karaktere token ID verme
      ↓
Karakter dizilerini modele gönderme
      ↓
Sonraki karakteri tahmin etme
      ↓
Yeni başkent benzeri isimler üretme
```

Karakter seviyesinde eğitim için kullanılan temel tokenizer:

```python
from tokenizer import CharTokenizer

tokenizer = CharTokenizer.from_file(DATA_FILE)
token_ids = tokenizer.encode(text)
```

TinyQwen eğitimi:

```bash
uv run python qwen3/train.py
```

TinyGemma eğitimi:

```bash
uv run python gemma4/train.py
```

Bu aşama sayesinde karakter tokenizasyonu, embedding katmanları, sonraki token tahmini ve autoregressive metin üretimi incelenmiştir.

---

# 2. BPE Tokenizer ile Metin Eğitimi

Karakter seviyesindeki çalışmadan sonra ikinci aşamada Byte Pair Encoding, yani BPE algoritması kullanılmıştır.

Bu aşamada evren, insanlık tarihi, bilim, teknoloji ve medeniyetler hakkında yazılmış uzun bir Türkçe metin kullanılmıştır.

Metin temizlendikten sonra tam 512 karakterlik bir eğitim parçası hazırlanmıştır.

---

## Metin Temizleme

Ham metin üzerinde şu işlemler uygulanmıştır:

- Unicode karakterler NFC biçiminde normalize edildi
- Görünmeyen karakterler temizlendi
- Fazladan boşluklar kaldırıldı
- Türkçe büyük harfler doğru biçimde küçültüldü
- Noktalama işaretleri standartlaştırıldı
- Tam 512 karakterlik eğitim metni oluşturuldu

Metni temizlemek için:

```bash
uv run python data/temizle_metin.py
```

Bu işlem sonucunda aşağıdaki dosyalar oluşturulur:

```text
data/temiz_metin.txt
data/temiz_metin_512.txt
```

---

## BPE Nedir?

Karakter tokenizer her karakteri bağımsız bir token olarak kullanırken BPE, metinde sık tekrar eden token çiftlerini birleştirir.

Örneğin başlangıçta:

```text
e | v | r | e | n
```

şeklinde parçalanan bir kelime, eğitim sırasında sık bulunan birleşmelere göre şu hale gelebilir:

```text
ev | ren
```

veya:

```text
evren
```

Böylece model daha uzun ve anlamlı metin parçalarıyla çalışabilir.

---

## BPE Tokenizer Eğitimi

BPE tokenizer şu komutla eğitilir:

```bash
uv run python train_bpe.py
```

Tokenizer eğitiminde kullanılan ayarlar:

| Özellik | Değer |
|---|---:|
| Eğitim metni | 512 karakter |
| Tokenizer türü | Byte-Level BPE |
| Vocabulary boyutu | 320 |
| BPE token sayısı | 313 |
| Özel tokenlar | `<unk>`, `<pad>`, `<bos>`, `<eos>` |

Eğitilen tokenizer şu dosyaya kaydedilir:

```text
data/bpe_tokenizer.json
```

512 karakterlik metin, BPE işleminden sonra 313 token ile temsil edilmiştir.

Bu sonuç, karakter sayısına göre daha kısa bir token dizisi elde edildiğini göstermektedir.

---

## Karakter Tokenizer ve BPE Karşılaştırması

| Özellik | Karakter Tokenizer | BPE Tokenizer |
|---|---|---|
| Temel birim | Tek karakter | Karakter veya alt kelime |
| Vocabulary | Küçük | Daha büyük |
| Token dizisi | Daha uzun | Genellikle daha kısa |
| Bilinmeyen kelime sorunu | Yok | Byte-Level kullanımında çok düşük |
| Öğrenilen yapılar | Harf geçişleri | Sık karakter ve kelime parçaları |
| Kullanılan veri | Başkent isimleri | 512 karakterlik Türkçe paragraf |

Karakter tokenizer daha basit bir yapı sunarken BPE, sık kullanılan metin parçalarını tek token altında birleştirebilir.

---

# TinyQwen BPE Eğitimi

BPE tokenları kullanılarak TinyQwen modeli şu komutla eğitilmiştir:

```bash
uv run python qwen3/train_bpe.py
```

## TinyQwen Sonuçları

| Özellik | Değer |
|---|---:|
| Eğitim metni | 512 karakter |
| BPE token sayısı | 313 |
| Vocabulary boyutu | 320 |
| Parametre sayısı | 28.864 |
| Block size | 64 |
| Eğitim adımı | 3000 |
| Başlangıç loss | 29.2655 |
| Son loss | 0.0237 |
| Rastgele tahmin loss değeri | 5.7683 |

TinyQwen tarafından üretilen örnek:

```text
evrenin sonsuzluğunda küçük bir damla olan insanlık, ilkel hücrelerden
yapay zekâ çağına dek büyüleyici ama bir o kadar da yıkıcı bir evrim
geçirmiştir. medeniyetler kurup uzayı keşfederken, kibrini yenemeyip
gezegenini tehlikeye atmıştır. yine
```

Eğitilen model şu dosyaya kaydedilmiştir:

```text
qwen3/tiny_qwen_bpe.pt
```

TinyQwen ile sonradan metin üretmek için:

```bash
uv run python qwen3/generate_bpe.py "evrenin" 150
```

---

# TinyGemma BPE Eğitimi

Aynı BPE tokenizer ve aynı eğitim metni TinyGemma modeli için de kullanılmıştır.

Model şu komutla eğitilmiştir:

```bash
uv run python gemma4/train_bpe.py
```

## TinyGemma Sonuçları

| Özellik | Değer |
|---|---:|
| Eğitim metni | 512 karakter |
| BPE token sayısı | 313 |
| Vocabulary boyutu | 320 |
| Block size | 64 |
| Eğitim adımı | 3000 |
| Son loss | 0.0247 |
| Rastgele tahmin loss değeri | 5.7683 |

TinyGemma tarafından üretilen örnek:

```text
evrenin sonsuzluğunda küçük bir damla olan insanlık, ilkel hücrelerden
yapay zekâ çağına dek büyüleyici ama bir o kadar da yıkıcı bir evrim
geçirmiştir. medeniyetler kurup uzayı keşfederken, kibrini yenemeyip
gezegenini tehlikeye atmıştır. yine
```

Eğitilen model şu dosyaya kaydedilmiştir:

```text
gemma4/tiny_gemma_bpe.pt
```

---

# Genel Çalışma Akışı

```text
Başkent isimleri
      ↓
Karakter tokenizer
      ↓
TinyQwen ve TinyGemma eğitimi
      ↓
Başkent benzeri isim üretimi
      ↓
Türkçe paragraf seçimi
      ↓
Metin temizleme
      ↓
512 karakterlik veri hazırlama
      ↓
BPE tokenizer eğitimi
      ↓
TinyQwen BPE eğitimi
      ↓
TinyGemma BPE eğitimi
      ↓
Model çıktılarının karşılaştırılması
```

---

# Proje Yapısı

```text
magibu-ilk-ders/
├── data/
│   ├── baskentler_bosluksuz.csv
│   ├── temiz_isimler.txt
│   ├── metin.txt
│   ├── temiz_metin.txt
│   ├── temiz_metin_512.txt
│   ├── bpe_tokenizer.json
│   ├── veri_temizleme.py
│   └── temizle_metin.py
│
├── qwen3/
│   ├── attention.py
│   ├── block.py
│   ├── config.py
│   ├── model.py
│   ├── tokenizer.py
│   ├── train.py
│   ├── train_bpe.py
│   ├── generate_bpe.py
│   ├── tiny_qwen.pt
│   └── tiny_qwen_bpe.pt
│
├── gemma4/
│   ├── attention.py
│   ├── block.py
│   ├── config.py
│   ├── model.py
│   ├── tokenizer.py
│   ├── train.py
│   ├── train_bpe.py
│   ├── tiny_gemma.pt
│   └── tiny_gemma_bpe.pt
│
├── train_bpe.py
├── demo.ipynb
├── pyproject.toml
└── uv.lock
```

Dosya adları mevcut projedeki isimlere göre küçük farklılıklar gösterebilir.

---

# Kurulum

Projede bağımlılık yönetimi için `uv` kullanılmaktadır.

```bash
uv sync
```

Gerekli paketleri elle eklemek için:

```bash
uv add torch tokenizers numpy
```

---

# Çalıştırma Sırası

## Karakter seviyesinde başkent eğitimi

Veriyi temizle:

```bash
uv run python data/veri_temizleme.py
```

TinyQwen modelini eğit:

```bash
uv run python qwen3/train.py
```

TinyGemma modelini eğit:

```bash
uv run python gemma4/train.py
```

## BPE eğitimi

Metni temizle:

```bash
uv run python data/temizle_metin.py
```

BPE tokenizer oluştur:

```bash
uv run python train_bpe.py
```

TinyQwen modelini BPE ile eğit:

```bash
uv run python qwen3/train_bpe.py
```

TinyGemma modelini BPE ile eğit:

```bash
uv run python gemma4/train_bpe.py
```

---

# Değerlendirme

Projenin ilk aşamasında karakter seviyesinde tokenizasyon kullanılarak başkent isimlerinin harf yapısı öğrenilmiştir. Modeller, bir önceki karaktere göre sonraki karakteri tahmin ederek başkent isimlerine benzeyen örnekler üretmiştir.

İkinci aşamada BPE tokenizer eğitilmiş ve metindeki sık tekrar eden karakter dizileri daha uzun tokenlar halinde birleştirilmiştir. Böylece 512 karakterlik metin 313 BPE tokenıyla temsil edilmiştir.

TinyQwen ve TinyGemma modellerinin loss değerleri eğitim sonunda oldukça düşük seviyelere ulaşmıştır. Bunun temel nedeni eğitim verisinin yalnızca 512 karakter olmasıdır. Modellerin bu kadar küçük bir veri kümesinde eğitim metnini büyük ölçüde ezberlemesi beklenen bir sonuçtur.

Bu çalışmanın amacı genel amaçlı bir Türkçe dil modeli geliştirmek değil; aşağıdaki konuları uygulamalı olarak incelemektir:

- Karakter seviyesinde tokenizasyon
- Byte Pair Encoding
- Vocabulary oluşturma
- Embedding katmanları
- Sonraki token tahmini
- Autoregressive metin üretimi
- TinyQwen ve TinyGemma mimarilerinin karşılaştırılması
- Eğitim loss değerlerinin incelenmesi

---

# Sonuç

Çalışmada önce karakter seviyesinde başkent üretme modeli hazırlanmış, ardından BPE tokenizer kullanılarak Türkçe metin üretimi gerçekleştirilmiştir.

Aynı veri ve tokenizer üzerinde iki farklı minyatür Transformer mimarisinin eğitilmesi sayesinde tokenizasyon yöntemleri ve model mimarileri karşılaştırmalı olarak incelenmiştir.
