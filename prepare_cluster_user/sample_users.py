import argparse
import json
import random
from pathlib import Path

from recbole.config import Config
from recbole.data import create_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DMF_CONFIG_DIR = PROJECT_ROOT / "prepare_cluster_user" / "DMF config"


def build_config(dataset_name, data_root):
    """Reuse the DMF configuration so the user table matches the CF model."""
    config_file = DMF_CONFIG_DIR / f"{dataset_name}.yaml"
    if not config_file.is_file():
        raise FileNotFoundError(f"DMF config not found: {config_file}")

    return Config(
        model="DMF",
        dataset=dataset_name,
        config_file_list=[str(config_file)],
        config_dict={"data_path": str(data_root) + "/"},
    )


def collect_internal_user_ids(dataset_name, data_root):
    """Return all internal user ids of the 5-core filtered dataset."""
    config = build_config(dataset_name, data_root)
    dataset = create_dataset(config)

    token2id = dataset.field2token_id["user_id"]
    return sorted(int(internal_id) for token, internal_id in token2id.items() if token != "[PAD]")


def sample_user_ids(dataset_name, data_root, sample_num, seed):
    all_ids = collect_internal_user_ids(dataset_name, data_root)
    if not all_ids:
        raise RuntimeError(f"No users found for dataset {dataset_name}")

    if sample_num >= len(all_ids):
        print(f"[warn] {dataset_name}: requested {sample_num} users, only {len(all_ids)} available")
        return all_ids

    rng = random.Random(seed)
    return rng.sample(all_ids, sample_num)


def main():
    parser = argparse.ArgumentParser(
        description="Sample a fixed evaluation user subset for each dataset."
    )
    parser.add_argument(
        "--dataset",
        nargs="+",
        required=True,
        help="one or more dataset names, e.g. Amazon_Arts Amazon_Games",
    )
    parser.add_argument(
        "--data_root",
        default=str(PROJECT_ROOT / "datasets"),
        help="directory that contains one sub-directory per dataset",
    )
    parser.add_argument(
        "--sample_num",
        type=int,
        default=1000,
        help="number of users to sample (also used in the output file name)",
    )
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument(
        "--output_dir",
        default=str(PROJECT_ROOT / "prepare_cluster_user" / "datasets"),
        help="root directory of the generated user lists",
    )
    args = parser.parse_args()

    for dataset_name in args.dataset:
        user_ids = sample_user_ids(dataset_name, args.data_root, args.sample_num, args.seed)

        output_dir = Path(args.output_dir) / dataset_name
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"users_{args.sample_num}.json"

        with open(output_path, "w") as f:
            json.dump([str(uid) for uid in user_ids], f)

        print(f"{dataset_name}: {len(user_ids)} users -> {output_path}")


if __name__ == "__main__":
    main()
