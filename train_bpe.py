from pathlib import Path
import json

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.normalizers import NFC
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer


TEXT_FILE = Path("data/temiz_metin_512.txt")
TOKENIZER_FILE = Path("data/bpe_tokenizer.json")

# Byte-level alfabe 256 temel sembol içerir.
# Dört özel token ile beraber 320, yaklaşık 60 yeni BPE birleşmesine
# izin veren küçük bir vocabulary oluşturur.
VOCAB_SIZE = 320
MIN_FREQUENCY = 2

SPECIAL_TOKENS = [
    "<unk>",
    "<pad>",
    "<bos>",
    "<eos>",
]


def main() -> None:
    if not TEXT_FILE.exists():
        raise FileNotFoundError(
            f"{TEXT_FILE} bulunamadı. Önce temizle_metin.py çalıştır."
        )

    text = TEXT_FILE.read_text(encoding="utf-8")

    if len(text) != 512:
        raise ValueError(
            f"Eğitim metni tam 512 karakter olmalı. Mevcut: {len(text)}"
        )

    # Boş bir BPE modeli oluştur.
    tokenizer = Tokenizer(
        BPE(unk_token="<unk>")
    )

    # Unicode gösterimini standartlaştır.
    tokenizer.normalizer = NFC()

    # Metni UTF-8 byte seviyesinde önceden parçala.
    tokenizer.pre_tokenizer = ByteLevel(
        add_prefix_space=False
    )

    # Token ID'lerini tekrar okunabilir metne çevirmek için.
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=MIN_FREQUENCY,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=ByteLevel.alphabet(),
        show_progress=True,
    )

    tokenizer.train(
        files=[str(TEXT_FILE)],
        trainer=trainer,
    )

    TOKENIZER_FILE.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(TOKENIZER_FILE))

    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded.ids)

    print("Tokenizer kaydedildi :", TOKENIZER_FILE)
    print("Vocabulary boyutu   :", tokenizer.get_vocab_size())
    print("Karakter sayısı      :", len(text))
    print("BPE token sayısı     :", len(encoded.ids))
    print("Decode doğru mu?     :", decoded == text)

    print("\nÖzel token ID'leri:")

    for token in SPECIAL_TOKENS:
        print(f"{token:6} -> {tokenizer.token_to_id(token)}")

    print("\nİlk 40 token:")
    print(encoded.tokens[:40])

    print("\nİlk 40 token ID:")
    print(encoded.ids[:40])

    # JSON içinden öğrenilen BPE merge kurallarını oku.
    tokenizer_config = json.loads(
        TOKENIZER_FILE.read_text(encoding="utf-8")
    )

    merges = tokenizer_config["model"].get("merges", [])

    print(f"\nÖğrenilen merge sayısı: {len(merges)}")
    print("\nİlk 20 BPE birleşmesi:")

    for number, merge in enumerate(merges[:20], start=1):
        if isinstance(merge, list):
            merge_text = " + ".join(merge)
        else:
            merge_text = str(merge)

        print(f"{number:2}. {merge_text}")


if __name__ == "__main__":
    main()