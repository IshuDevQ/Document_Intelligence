import shutil
from pathlib import Path

import kagglehub


DATASET_NAME = "divg07/casia-20-image-tampering-detection-dataset"

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}


def find_folder(root: Path, folder_name: str) -> Path | None:
    for path in root.rglob("*"):
        if path.is_dir() and path.name.lower() == folder_name.lower():
            return path

    return None


def copy_images(source_folder: Path, target_folder: Path) -> int:
    target_folder.mkdir(parents=True, exist_ok=True)

    count = 0

    for image_path in source_folder.rglob("*"):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        target_path = target_folder / image_path.name

        if target_path.exists():
            target_path = target_folder / f"{image_path.stem}_{count}{image_path.suffix}"

        shutil.copy2(image_path, target_path)
        count += 1

    return count


def main():
    print("Downloading CASIA 2.0 dataset from Kaggle...")

    downloaded_path = kagglehub.dataset_download(DATASET_NAME)
    dataset_root = Path(downloaded_path)

    print("Dataset downloaded at:")
    print(dataset_root)

    project_root = Path(__file__).resolve().parent

    authentic_target = project_root / "data" / "training_set" / "authentic"
    tampered_target = project_root / "data" / "training_set" / "tampered"

    au_folder = find_folder(dataset_root, "Au")
    tp_folder = find_folder(dataset_root, "Tp")

    if au_folder is None:
        raise FileNotFoundError(
            "Could not find the Au folder. Please check the downloaded dataset structure."
        )

    if tp_folder is None:
        raise FileNotFoundError(
            "Could not find the Tp folder. Please check the downloaded dataset structure."
        )

    print("Authentic folder found:", au_folder)
    print("Tampered folder found:", tp_folder)

    authentic_count = copy_images(au_folder, authentic_target)
    tampered_count = copy_images(tp_folder, tampered_target)

    print()
    print("Dataset organized successfully.")
    print("Authentic images copied:", authentic_count)
    print("Tampered images copied:", tampered_count)
    print("Authentic target:", authentic_target)
    print("Tampered target:", tampered_target)


if __name__ == "__main__":
    main()