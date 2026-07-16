"""BPE tokenizer ile TinyGemma modelini eğitir.

Proje ana dizininden çalıştır:

    uv run python gemma4/train_bpe.py
"""

from pathlib import Path
import math

import torch
from tokenizers import Tokenizer

from config import ModelConfig
from model import TinyGemma


# ---------------------------------------------------------------------------
# Dosya yolları
# ---------------------------------------------------------------------------

GEMMA_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = GEMMA_DIR.parent

DATA_FILE = PROJECT_ROOT / "data" / "temiz_metin_512.txt"
TOKENIZER_FILE = PROJECT_ROOT / "data" / "bpe_tokenizer.json"
CHECKPOINT_FILE = GEMMA_DIR / "tiny_gemma_bpe.pt"


# ---------------------------------------------------------------------------
# Eğitim ayarları
# ---------------------------------------------------------------------------

BATCH_SIZE = 32
BLOCK_SIZE = 64

STEPS = 3000
LEARNING_RATE = 3e-3
EVAL_EVERY = 200
SEED = 1337


def select_device() -> str:
    """Kullanılabilecek en uygun PyTorch cihazını seçer."""

    if torch.cuda.is_available():
        return "cuda"

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return "mps"

    return "cpu"


device = select_device()
torch.manual_seed(SEED)


# ---------------------------------------------------------------------------
# Dosya kontrolleri
# ---------------------------------------------------------------------------

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Eğitim metni bulunamadı:\n{DATA_FILE}"
    )

if not TOKENIZER_FILE.exists():
    raise FileNotFoundError(
        f"BPE tokenizer bulunamadı:\n{TOKENIZER_FILE}\n"
        "Önce train_bpe.py dosyasını çalıştır."
    )


# ---------------------------------------------------------------------------
# BPE tokenizer ve eğitim verisi
# ---------------------------------------------------------------------------

tokenizer = Tokenizer.from_file(
    str(TOKENIZER_FILE)
)

vocab_size = tokenizer.get_vocab_size()

text = DATA_FILE.read_text(
    encoding="utf-8"
)

token_ids = tokenizer.encode(text).ids

data = torch.tensor(
    token_ids,
    dtype=torch.long,
)

print(f"Metindeki karakter sayısı : {len(text)}")
print(f"BPE token sayısı          : {len(token_ids)}")
print(f"Vocabulary boyutu         : {vocab_size}")


if len(text) != 512:
    raise ValueError(
        f"Metin tam 512 karakter olmalı. "
        f"Mevcut uzunluk: {len(text)}"
    )

if len(data) <= BLOCK_SIZE + 1:
    raise ValueError(
        "Eğitim verisi BLOCK_SIZE için yetersiz.\n"
        f"Token sayısı: {len(data)}\n"
        f"BLOCK_SIZE: {BLOCK_SIZE}"
    )


# ---------------------------------------------------------------------------
# Batch oluşturma
# ---------------------------------------------------------------------------

def get_batch() -> tuple[torch.Tensor, torch.Tensor]:
    """Rastgele bağlam ve hedef token dizileri üretir."""

    max_start = len(data) - BLOCK_SIZE - 1

    indices = torch.randint(
        low=0,
        high=max_start + 1,
        size=(BATCH_SIZE,),
    )

    starts = indices.tolist()

    x = torch.stack(
        [
            data[start:start + BLOCK_SIZE]
            for start in starts
        ]
    )

    y = torch.stack(
        [
            data[start + 1:start + BLOCK_SIZE + 1]
            for start in starts
        ]
    )

    return x.to(device), y.to(device)


# ---------------------------------------------------------------------------
# TinyGemma modeli
# ---------------------------------------------------------------------------

cfg = ModelConfig(
    vocab_size=vocab_size,
    max_seq_len=BLOCK_SIZE,
)

model = TinyGemma(cfg).to(device)

number_of_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)

layer_types = [
    "global" if is_global else "local"
    for is_global in model.layer_is_global
]

print(f"device                     : {device}")
print(f"parameters                 : {number_of_parameters:,}")
print(f"block size                 : {BLOCK_SIZE}")
print(f"layers                     : {layer_types}")

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
)


# ---------------------------------------------------------------------------
# Metin üretme fonksiyonu
# ---------------------------------------------------------------------------

def sample_text(
    prompt: str = "evrenin",
    max_new_tokens: int = 150,
    temperature: float = 0.8,
) -> str:
    """Prompt ile başlayan örnek bir metin üretir."""

    model.eval()

    prompt_ids = tokenizer.encode(prompt).ids

    if not prompt_ids:
        raise ValueError(
            f"Prompt tokenlaştırılamadı: {prompt!r}"
        )

    # Bağlam sınırını aşmamak için son tokenları kullan.
    prompt_ids = prompt_ids[-BLOCK_SIZE:]

    input_ids = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
        device=device,
    )

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=40,
            eos_id=None,
        )

    generated_text = tokenizer.decode(
        output_ids[0].tolist(),
        skip_special_tokens=True,
    )

    model.train()

    return generated_text


# ---------------------------------------------------------------------------
# Eğitim döngüsü
# ---------------------------------------------------------------------------

model.train()

for step in range(1, STEPS + 1):
    x, y = get_batch()

    _, loss = model(x, y)

    optimizer.zero_grad(
        set_to_none=True
    )

    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=1.0,
    )

    optimizer.step()

    if step == 1 or step % EVAL_EVERY == 0:
        print(
            f"step {step:5d} | loss {loss.item():.4f}"
        )


# ---------------------------------------------------------------------------
# Sonuç ve örnek üretim
# ---------------------------------------------------------------------------

baseline_loss = math.log(vocab_size)

print(
    "\nBaseline loss (uniform guessing): "
    f"{baseline_loss:.4f}"
)

print("\nÜretilen örnek:\n")

generated_example = sample_text(
    prompt="evrenin",
    max_new_tokens=150,
    temperature=0.8,
)

print(generated_example)


# ---------------------------------------------------------------------------
# Checkpoint kaydetme
# ---------------------------------------------------------------------------

torch.save(
    {
        "model": model.state_dict(),
        "cfg": cfg,
        "tokenizer_file": "data/bpe_tokenizer.json",
        "data_file": "data/temiz_metin_512.txt",
        "vocab_size": vocab_size,
        "block_size": BLOCK_SIZE,
        "steps": STEPS,
        "final_loss": loss.item(),
        "generated_example": generated_example,
        "architecture": "TinyGemma",
    },
    CHECKPOINT_FILE,
)

print(
    f"\nCheckpoint kaydedildi:\n{CHECKPOINT_FILE}"
)