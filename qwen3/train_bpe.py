"""BPE tokenizer ile TinyQwen modelini eğit.

Proje kökünden çalıştır:

    uv run python qwen3/train.py
"""

from pathlib import Path

import torch
from tokenizers import Tokenizer

from config import ModelConfig
from model import TinyQwen


# ---------------------------------------------------------------------------
# Dosya yolları
# ---------------------------------------------------------------------------

QWEN_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = QWEN_DIR.parent

DATA_FILE = PROJECT_ROOT / "data" / "temiz_metin_512.txt"
TOKENIZER_FILE = PROJECT_ROOT / "data" / "bpe_tokenizer.json"
CHECKPOINT_FILE = QWEN_DIR / "tiny_qwen_bpe.pt"


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

BATCH_SIZE = 32

# BPE kullandığımız için 16 çok kısa kalabilir.
BLOCK_SIZE = 64

STEPS = 3000
LEARNING_RATE = 3e-3
EVAL_EVERY = 200
SEED = 1337

device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(SEED)


# ---------------------------------------------------------------------------
# Dosya kontrolleri
# ---------------------------------------------------------------------------

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Eğitim metni bulunamadı:\n{DATA_FILE}\n"
        "Önce temizle_metin.py dosyasını çalıştır."
    )

if not TOKENIZER_FILE.exists():
    raise FileNotFoundError(
        f"BPE tokenizer bulunamadı:\n{TOKENIZER_FILE}\n"
        "Önce train_bpe.py dosyasını çalıştır."
    )


# ---------------------------------------------------------------------------
# BPE tokenizer
# ---------------------------------------------------------------------------

tokenizer = Tokenizer.from_file(str(TOKENIZER_FILE))

vocab_size = tokenizer.get_vocab_size()

text = DATA_FILE.read_text(encoding="utf-8")

encoding = tokenizer.encode(text)
token_ids = encoding.ids

data = torch.tensor(
    token_ids,
    dtype=torch.long,
)

print(f"Metindeki karakter sayısı : {len(text)}")
print(f"BPE token sayısı          : {len(token_ids)}")
print(f"Vocabulary boyutu         : {vocab_size}")

if len(data) <= BLOCK_SIZE + 1:
    raise ValueError(
        f"Eğitim verisi çok kısa.\n"
        f"Token sayısı: {len(data)}\n"
        f"BLOCK_SIZE: {BLOCK_SIZE}\n\n"
        f"BLOCK_SIZE değerini {len(data) - 2} veya daha küçük yap."
    )


# ---------------------------------------------------------------------------
# Batch oluşturma
# ---------------------------------------------------------------------------

def get_batch() -> tuple[torch.Tensor, torch.Tensor]:
    """Rastgele metin parçaları seçer.

    x: giriş tokenları
    y: bir token sağa kaydırılmış hedef tokenlar
    """

    max_start = len(data) - BLOCK_SIZE - 1

    indices = torch.randint(
        low=0,
        high=max_start + 1,
        size=(BATCH_SIZE,),
    )

    x = torch.stack(
        [
            data[index:index + BLOCK_SIZE]
            for index in indices.tolist()
        ]
    )

    y = torch.stack(
        [
            data[index + 1:index + 1 + BLOCK_SIZE]
            for index in indices.tolist()
        ]
    )

    return x.to(device), y.to(device)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

cfg = ModelConfig(
    vocab_size=vocab_size,
    max_seq_len=BLOCK_SIZE,
)

model = TinyQwen(cfg).to(device)

n_params = sum(
    parameter.numel()
    for parameter in model.parameters()
)

print(f"device                     : {device}")
print(f"parameters                 : {n_params:,}")
print(f"block size                 : {BLOCK_SIZE}")


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
)


# ---------------------------------------------------------------------------
# Metin üretme
# ---------------------------------------------------------------------------

def sample_text(
    prompt: str = "evrenin",
    max_new_tokens: int = 100,
) -> str:
    """Verilen başlangıç metninden devam üretir."""

    model.eval()

    prompt_ids = tokenizer.encode(prompt).ids

    if not prompt_ids:
        raise ValueError(
            f"Prompt tokenlaştırılamadı: {prompt!r}"
        )

    # Prompt modelin maksimum bağlamından uzunsa son kısmını kullan.
    prompt_ids = prompt_ids[-BLOCK_SIZE:]

    start = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
        device=device,
    )

    with torch.no_grad():
        output = model.generate(
            start,
            max_new_tokens=max_new_tokens,
            temperature=0.8,
            top_k=40,
            eos_id=None,
        )

    generated_ids = output[0].tolist()

    generated_text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    )

    model.train()

    return generated_text


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

model.train()

for step in range(1, STEPS + 1):
    x, y = get_batch()

    _, loss = model(x, y)

    optimizer.zero_grad(set_to_none=True)

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
# Eğitim sonrası sonuç
# ---------------------------------------------------------------------------

baseline_loss = torch.log(
    torch.tensor(float(vocab_size))
)

print(
    "\nBaseline loss (uniform guessing): "
    f"{baseline_loss.item():.4f}"
)

print("\nÜretilen örnek:\n")

print(
    sample_text(
        prompt="evrenin",
        max_new_tokens=150,
    )
)


# ---------------------------------------------------------------------------
# Checkpoint kaydetme
# ---------------------------------------------------------------------------

torch.save(
    {
        "model": model.state_dict(),
        "cfg": cfg,
        "tokenizer_file": str(TOKENIZER_FILE),
        "data_file": str(DATA_FILE),
        "vocab_size": vocab_size,
        "block_size": BLOCK_SIZE,
        "steps": STEPS,
    },
    CHECKPOINT_FILE,
)

print(
    f"\nCheckpoint kaydedildi:\n{CHECKPOINT_FILE}"
)