from huggingface_hub import snapshot_download
import fire
import os
import logging
import glob
import tarfile

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main(
    repo_id="mozilla-foundation/common_voice_17_0",
    output_dir="/mnt/home/giuseppe/myscratch/fair-asr-leaderboard/data",
):
    local_dir = repo_id.replace("/", "--")

    patterns = list()
    for lang in LANGS:
        for split in SPLITS:
            patterns.append(f"audio/{lang}/{split}/*.tar")
            patterns.append(f"transcript/{lang}/{split}.tsv")

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=os.path.join(output_dir, local_dir),
        allow_patterns=patterns,
    )

    tar_files = []
    for pattern in patterns:
        tar_files.extend(glob.glob(os.path.join(output_dir, local_dir, pattern)))

    for tar_file in tar_files:
        if tarfile.is_tarfile(tar_file):
            with tarfile.open(tar_file, "r") as tar:
                tar.extractall(path=os.path.dirname(tar_file))
                logger.info(f"Extracted {tar_file}")


if __name__ == "__main__":
    fire.Fire(main)
