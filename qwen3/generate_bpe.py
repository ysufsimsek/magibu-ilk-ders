from pathlib import Path
import sys

import torch
from tokenizers import Tokenizer

from model import TinyQwen


QWEN_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = QWEN_DIR.parent

CHECKPOINT_FILE = QWEN_DIR / "tiny_qwen_bpe.pt"
TOKENIZER_FILE = PROJECT_ROOT / "data" / "bpe_tokenizer.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main() -> None:
    prompt = sys.argv[1] if len(sys.argv) > 1 else "evrenin"
    max_new_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 100

    tokenizer = Tokenizer.from_file(str(TOKENIZER_FILE))

    checkpoint = torch.load(
        CHECKPOINT_FILE,
        map_location=DEVICE,
        weights_only=False,
    )

    model = TinyQwen(checkpoint["cfg"]).to(DEVICE)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    prompt_ids = tokenizer.encode(prompt).ids

    if not prompt_ids:
        raise ValueError("Prompt tokenlaştırılamadı.")

    input_ids = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
        device=DEVICE,
    )

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.8,
            top_k=40,
            eos_id=None,
        )

    generated_text = tokenizer.decode(
        output_ids[0].tolist(),
        skip_special_tokens=True,
    )

    print(generated_text)


if __name__ == "__main__":
    main()