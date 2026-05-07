"""Load Kaggle / ImageFolder data for latent-diffusion training."""
from __future__ import annotations

import csv
from pathlib import Path

import kagglehub
from PIL import Image
from PIL import ImageFile
from torch.utils.data import Dataset
from torchvision import datasets, transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True


KAGGLE_SLUG = "ashishjangra27/image-classification-200k-dataset"


def download_kaggle_dataset() -> Path:
    """Download dataset via kagglehub and return root directory (contains data.csv + images)."""
    return Path(kagglehub.dataset_download(KAGGLE_SLUG))


def resolve_version_root(base: Path) -> Path:
    """If hub returned repo root, descend to versions/N where data.csv lives."""
    base = base.resolve()
    if (base / "data.csv").is_file():
        return base
    versions = base / "versions"
    if versions.is_dir():
        subs = sorted(int(x.name) for x in versions.iterdir() if x.is_dir() and x.name.isdigit())
        if subs:
            cand = versions / str(subs[-1])
            if (cand / "data.csv").is_file():
                return cand.resolve()
    raise FileNotFoundError(f"No data.csv under {base}")


def _has_class_subdirs(path: Path, min_folders: int = 2) -> bool:
    if not path.is_dir():
        return False
    subs = [p for p in path.iterdir() if p.is_dir()]
    return len(subs) >= min_folders


def discover_imagefolder_root(base: Path) -> Path:
    """Find folder suitable for torchvision.datasets.ImageFolder (class subdirs)."""
    for name in ("train", "Train", "TRAIN", "training"):
        p = base / name
        if _has_class_subdirs(p):
            return p.resolve()
    if _has_class_subdirs(base):
        return base.resolve()
    best: Path | None = None
    best_count = 0
    for sub in base.rglob("*"):
        if sub.is_dir() and _has_class_subdirs(sub):
            n = len([x for x in sub.iterdir() if x.is_dir()])
            if n > best_count:
                best_count = n
                best = sub
    if best is not None:
        return best.resolve()
    raise FileNotFoundError(
        f"Could not find a subdirectory with class folders under {base}. "
        "Use a CSV-based dataset (data.csv) or pass --data_root to ImageFolder train/."
    )


def resolve_csv_image_path(base: Path, csv_rel_path: str) -> Path | None:
    """Map CSV path column to an existing file (handles Imgs vs Imgs/Imgs layouts)."""
    rel = csv_rel_path.strip().replace("\\", "/")
    candidates = [
        base / rel,
        base / "Imgs" / rel,
        base / rel.replace("Imgs/", "Imgs/Imgs/", 1) if rel.startswith("Imgs/") else None,
        base / "Imgs" / "Imgs" / Path(rel).name,
    ]
    for c in candidates:
        if c is not None and c.is_file():
            return c
    return None


class CsvTagsDataset(Dataset):
    """Reads ashishjangra27 CSV: image path + tags column used as text prompt."""

    def __init__(self, version_root: Path, image_size: int):
        super().__init__()
        self.root = version_root.resolve()
        csv_path = self.root / "data.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(csv_path)

        tfm = build_transform(image_size)
        self.rows: list[tuple[Path, str]] = []
        with open(csv_path, newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rel = row.get("path") or row.get("Path")
                tags = (row.get("tags") or row.get("Tags") or "").strip()
                if not rel or not tags:
                    continue
                p = resolve_csv_image_path(self.root, rel)
                if p is not None:
                    self.rows.append((p, tags))

        if not self.rows:
            raise RuntimeError(f"No valid image paths parsed from {csv_path}")

        self.transform = tfm
        self._bad_indices: set[int] = set()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        max_tries = 20
        idx = int(index)
        last_error: Exception | None = None

        for _ in range(max_tries):
            path, tags = self.rows[idx]
            if idx in self._bad_indices:
                idx = (idx + 1) % len(self.rows)
                continue
            try:
                with Image.open(path) as img:
                    image = img.convert("RGB")
                return self.transform(image), tags
            except Exception as exc:
                self._bad_indices.add(idx)
                last_error = exc
                idx = (idx + 1) % len(self.rows)

        raise RuntimeError(
            "Failed to read images after multiple retries; too many corrupt files in sampled window."
        ) from last_error


class ImageFolderPromptDataset(Dataset):
    """ImageFolder with string prompts from folder names."""

    def __init__(self, folder: Path | str, image_size: int, prompt_template: str):
        super().__init__()
        self.folder = Path(folder)
        self.prompt_template = prompt_template
        self.ds = datasets.ImageFolder(str(self.folder), transform=build_transform(image_size))

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, index: int):
        x, y = self.ds[index]
        name = self.ds.classes[y]
        prompt = self.prompt_template.format(name=name)
        return x, prompt


def build_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )


def load_training_dataset(
    data_root: Path,
    image_size: int,
    prompt_template: str,
) -> Dataset:
    """
    Prefer CSV layout (data.csv + tags). Otherwise ImageFolder under data_root.
    """
    root = data_root.resolve()
    if (root / "data.csv").is_file():
        return CsvTagsDataset(root, image_size)
    img_root = discover_imagefolder_root(root)
    return ImageFolderPromptDataset(img_root, image_size, prompt_template)

