from pathlib import Path
import re
import unicodedata


DATA_DIR = Path(__file__).resolve().parent

INPUT_FILE = DATA_DIR / "metin.txt"
CLEAN_FILE = DATA_DIR / "temiz_metin.txt"
SAMPLE_FILE = DATA_DIR / "temiz_metin_512.txt"

TARGET_LENGTH = 512
LOWERCASE = True


def turkish_lower(text: str) -> str:
    """Türkçedeki I/İ harflerini doğru biçimde küçültür."""
    return text.replace("I", "ı").replace("İ", "i").lower()


def clean_text(text: str) -> str:
    # Aynı görünen Unicode karakterleri aynı biçime getirir.
    text = unicodedata.normalize("NFC", text)

    # BOM ve bazı tipografik karakterleri standartlaştırır.
    text = text.replace("\ufeff", "")

    text = text.translate(
        str.maketrans(
            {
                "\u00a0": " ",  # bölünemez boşluk
                "“": '"',
                "”": '"',
                "„": '"',
                "’": "'",
                "‘": "'",
                "–": "-",
                "—": "-",
                "…": "...",
            }
        )
    )

    # Görünmeyen kontrol karakterlerini boşluk yapar.
    text = "".join(
        " " if unicodedata.category(char).startswith("C") else char
        for char in text
    )

    # Tekrarlanan boşluk ve satır sonlarını tek boşluğa indirir.
    text = re.sub(r"\s+", " ", text).strip()

    if LOWERCASE:
        text = turkish_lower(text)

    return text


def is_end_boundary(text: str, end: int) -> bool:
    """512. karakterin bir kelimenin ortasına denk gelmesini engeller."""
    return end == len(text) or text[end].isspace()


def extract_exact_window(text: str, size: int) -> tuple[str, int]:
    if len(text) < size:
        raise ValueError(
            f"Temiz metin en az {size} karakter olmalı. "
            f"Mevcut uzunluk: {len(text)}"
        )

    # Önce cümle başından başlayan uygun bir pencere ara.
    sentence_starts = [0]

    for match in re.finditer(r"(?<=[.!?])\s+", text):
        sentence_starts.append(match.end())

    for start in sentence_starts:
        end = start + size

        if end <= len(text) and is_end_boundary(text, end):
            return text[start:end], start

    # Cümle başlangıcı bulunamazsa herhangi bir kelime sınırını kullan.
    for start in range(len(text) - size + 1):
        starts_at_boundary = start == 0 or text[start - 1].isspace()

        if not starts_at_boundary:
            continue

        end = start + size

        if is_end_boundary(text, end):
            return text[start:end], start

    # Çok sıra dışı durumda ilk 512 karakteri al.
    return text[:size], 0


def write_without_extra_newline(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as file:
        file.write(text)


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"{INPUT_FILE} bulunamadı. Metni bu dosyaya kaydet."
        )

    raw_text = INPUT_FILE.read_text(encoding="utf-8")
    cleaned_text = clean_text(raw_text)

    sample, start_index = extract_exact_window(
        cleaned_text,
        TARGET_LENGTH,
    )

    write_without_extra_newline(CLEAN_FILE, cleaned_text)
    write_without_extra_newline(SAMPLE_FILE, sample)

    print(f"Ham karakter sayısı       : {len(raw_text)}")
    print(f"Temiz karakter sayısı     : {len(cleaned_text)}")
    print(f"512'lik parçanın başlangıcı: {start_index}")
    print(f"Eğitim metni uzunluğu     : {len(sample)}")
    print(f"UTF-8 byte sayısı         : {len(sample.encode('utf-8'))}")
    print(f"Tam metin                 : {CLEAN_FILE}")
    print(f"512 karakterlik metin     : {SAMPLE_FILE}")

    assert len(sample) == TARGET_LENGTH

    print("\n512 karakterlik eğitim metni:\n")
    print(sample)


if __name__ == "__main__":
    main()