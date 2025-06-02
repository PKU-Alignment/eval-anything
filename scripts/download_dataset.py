import argparse
import logging

from huggingface_hub import hf_hub_download


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_datasets(repo_id, files, local_dir):
    for file_name in files:
        try:
            file_path = hf_hub_download(
                repo_id=repo_id,
                filename=file_name,
                repo_type='dataset',
                local_dir=local_dir,
                local_dir_use_symlinks=False,
            )
            logger.info(f'Downloaded {file_name} to {file_path}')
        except Exception as e:
            logger.error(f'Failed to download {file_name}: {e}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download datasets from Hugging Face Hub.')
    parser.add_argument(
        '--save_dir',
        type=str,
        default='.',
        help='Local directory to save the downloaded files (default: current directory)',
    )

    args = parser.parse_args()

    download_datasets(
        'SafetyEmbodiedAI/safety-chores',
        ['fetchtype.jsonl.gz', 'objectnavtype.jsonl.gz', 'pickuptype.jsonl.gz'],
        args.save_dir,
    )
