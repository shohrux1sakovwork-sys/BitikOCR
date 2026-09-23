import os
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import HfApi, get_token, hf_hub_download
from tqdm.auto import tqdm

# Directory where images will be stored flat (no part1/part2)
os.makedirs("data/images", exist_ok=True)

# Get all image file paths from repository
api = HfApi(token=get_token())
info = api.dataset_info("aktrmai/synthetic_mock")
image_files = [
    s.rfilename for s in info.siblings
    if s.rfilename.startswith("images/") and s.rfilename.endswith(".jpg")
]

def download_image(repo_file_path: str) -> None:
    filename = os.path.basename(repo_file_path)
    dest = os.path.join("data", "images", filename)
    if os.path.exists(dest):
        return
    cached = hf_hub_download(
        repo_id="aktrmai/synthetic_mock",
        repo_type="dataset",
        filename=repo_file_path,
    )
    # Save flat into data/images/<filename> without part1/part2
    real_path = os.path.realpath(cached)
    if not os.path.exists(dest):
        try:
            os.symlink(real_path, dest)
        except OSError:
            import shutil
            shutil.copyfile(real_path, dest)

print(f"Downloading {len(image_files)} images into data/images/...")
with ThreadPoolExecutor(max_workers=16) as executor:
    list(tqdm(executor.map(download_image, image_files), total=len(image_files), desc="Images"))

print("Done! Total images in data/images:", len(os.listdir("data/images")))