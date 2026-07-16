from pathlib import Path
import sys
import traceback

import torch


ROOT = Path(__file__).resolve().parent
QWEN_DIR = ROOT / "qwen3"

sys.path.insert(0, str(QWEN_DIR))

from model import TinyQwen  # noqa: E402


def main() -> None:
    checkpoint_path = QWEN_DIR / "tiny_qwen.pt"

    print("Python:", sys.executable, flush=True)
    print("PyTorch:", torch.__version__, flush=True)
    print("Checkpoint:", checkpoint_path, flush=True)
    print("Checkpoint var mı?:", checkpoint_path.exists(), flush=True)

    ckpt = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    print("\nCheckpoint anahtarları:", ckpt.keys(), flush=True)

    model = TinyQwen(ckpt["cfg"])
    model.load_state_dict(ckpt["model"])
    model.eval()

    print("\nMODEL TÜRÜ")
    print(type(model), flush=True)

    print("\nMODEL YAPISI")
    print(model, flush=True)

    print("\nÜST SEVİYE KATMANLAR")
    for name, layer in model.named_children():
        print(f"{name}: {type(layer).__name__}", flush=True)

    print("\nEMBEDDING KATMANLARI")
    embedding_found = False

    for name, layer in model.named_modules():
        if isinstance(layer, torch.nn.Embedding):
            embedding_found = True
            print(
                f"{name}: {type(layer).__name__}, "
                f"shape={tuple(layer.weight.shape)}",
                flush=True,
            )

    if not embedding_found:
        print("nn.Embedding katmanı bulunamadı.", flush=True)

    print("\nEMBED/TOKEN İÇEREN PARAMETRELER")
    matching_parameter_found = False

    for name, parameter in model.named_parameters():
        lowered_name = name.lower()

        if "embed" in lowered_name or "token" in lowered_name:
            matching_parameter_found = True
            print(f"{name}: {tuple(parameter.shape)}", flush=True)

    if not matching_parameter_found:
        print("Embed veya token isimli parametre bulunamadı.", flush=True)

    print("\nMODELDE 'embed' ALANI VAR MI?")
    print("hasattr(model, 'embed'):", hasattr(model, "embed"), flush=True)

    if hasattr(model, "embed"):
        embed = getattr(model, "embed")
        print("model.embed türü:", type(embed), flush=True)
        print("model.embed:", embed, flush=True)

        if hasattr(embed, "weight"):
            print(
                "model.embed.weight.shape:",
                tuple(embed.weight.shape),
                flush=True,
            )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\nPROGRAM HATA VERDİ:", flush=True)
        traceback.print_exc()
        raise