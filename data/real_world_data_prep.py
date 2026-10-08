import argparse
import os
import pickle
import random
import re

# TODO: generalize the numbver of location
# TODO: user 2 ND USER 6 REMOVED FOR NOW . dont have data for all 7 locations
import zipfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

LOCATIONS = [
    "chest",
    # "forearm", # missing climbingup for participant 2 
    "head",
    "shin",
    # "thigh", # missing jumping for participant 6
    "upperarm",
    "waist",
]


def unzip_all(root: str) -> None:
    """
    Walk `root` recursively, find *.zip files and extract each one into a
    folder with the same name minus the `.zip` suffix.

    Example
    -------
    >>> unzip_all("/path/to/realworld2016_dataset")
    """
    root_path = Path(root)

    for z_path in root_path.rglob("*.zip"):
        dest_dir = z_path.with_suffix("")  # drop the '.zip'
        dest_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(z_path) as zf:
            zf.extractall(dest_dir)

        print(
            f"✓ extracted {z_path.relative_to(root_path)} → {dest_dir.relative_to(root_path)}"
        )


def parse_arguments():
    parser = argparse.ArgumentParser(description="Parameters for preparing RWHAR")
    parser.add_argument(
        "--dataset_loc",
        type=str,
        default="data/raw/rwhar",
        help="Location of the raw sensory data (see docs/DATASETS.md)",
    )
    parser.add_argument(
        "--sampling_rate",
        type=int,
        default=50,
        help="Sampling rate for the data. Is used to downsample to the required rate",
    )
    parser.add_argument(
        "--original_sampling_rate",
        type=int,
        default=50,
        help="Original sampling rate for the dataset",
    )
    parser.add_argument(
        "--perform_normalization",
        type=str,
        default="True",
        help="To perform mean-variance normalization on the data",
    )
    parser.add_argument(
        "--num_sensor_channels",
        type=int,
        default=3,
        help="Number of sensor channels for used in the data preparation",
    )
    parser.add_argument(
        "--n_fold_validation",
        type=int,
        default=5,
        help="To extract data with n-folds instead of random "
        "20% test set. Default is 0, which creates the "
        "normal 80-20 split.",
    )
    parser.add_argument(
        "--null_class",
        type=str,
        default="False",
        help="To move all the transitionary classes to NULL",
    )

    parser.add_argument("--seed", type=int, default=42, help="random seed value")
    parser.add_argument("--loc", type=str, default="waist", help="Sensor location")
    parser.add_argument(
        "--scaler_path",
        type=str,
        default=None,
        help="Standard scaler already fit on some dataset to be used for normalising current dataset",
    )

    args = parser.parse_args()

    return args


def map_activity_to_id():
    activity_list = [
        "climbingdown", "climbingup", "jumping", "lying",
        "running", "sitting", "standing", "walking",
    ]
    return {act: idx for idx, act in enumerate(activity_list)}, activity_list


def perform_train_val_test_split(unique_subj, test_size=0.2, val_size=0.2, seed=42):
    # Doing the train-test split
    train_val_subj, test_subj = train_test_split(
        unique_subj, test_size=test_size, random_state=seed
    )
    print("The train and validation subjects are: {}".format(train_val_subj))
    print("The test subjects are: {}".format(test_subj))

    # Splitting further into train and validation subjects
    train_subj, val_subj = train_test_split(
        train_val_subj, test_size=val_size, random_state=seed
    )

    subjects = {"train": train_subj, "val": val_subj, "test": test_subj}

    folder = os.path.join("all_data", date.today().strftime("%b-%d-%Y"))
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "subjects.pkl"), "wb") as f:
        pickle.dump(subjects, f, pickle.HIGHEST_PROTOCOL)

    return subjects


def downsample_data(data):
    return data


def get_data(args):
    act_to_idx, activity_names = map_activity_to_id()

    # user_data[user][activity] = ndarray(T, C)
    user_data = {user_no: {} for user_no in range(1, 16)}

    missing_locs = []

    # iterate over subjects
    for participant_num in range(1, 16):
        print("user", participant_num)
        user_folder = Path(args.dataset_loc) / f"proband{participant_num}" / "data"

        # iterate over activities
        for activity_dir in user_folder.glob("acc_*_csv"):
            activity = activity_dir.name[4:-4]  # strip 'acc_' and '_csv'
            if activity not in act_to_idx:
                continue
            activity_idx = act_to_idx[activity]

            loc_list = LOCATIONS if args.loc == "all" else [args.loc]
            per_loc_data = []

            for loc in loc_list:
                pattern = re.compile(f"acc_{activity}(_\\d+)?_{loc}\.csv")
                csv_files = sorted(
                    p for p in activity_dir.rglob("*.csv") if pattern.search(p.name)
                )
                if not csv_files:
                    missing_locs.append((participant_num, activity, loc))
                    print('Data missing for ', (participant_num, activity, loc))
                    continue

                # aggregate all segments for that (user, activity, loc)
                segs = []
                for csv_file in csv_files:
                    df = pd.read_csv(csv_file)
                    segs.append(df[["attr_x", "attr_y", "attr_z"]].values)
                seg = np.vstack(segs)  # (T,3)
                per_loc_data.append(seg)

            if not per_loc_data:
                continue  # nothing found for this (user, activity)

            # Ensure all locations have the same length; trim to min length
            min_len = min(d.shape[0] for d in per_loc_data)
            per_loc_data = [d[:min_len] for d in per_loc_data]

            concatenated = np.hstack(per_loc_data)  # (T, 3*len(loc_list))
            user_data[participant_num][activity_idx] = concatenated

    # Build one big DataFrame
    loc_list = LOCATIONS if args.loc == "all" else [args.loc]
    columns = (
        [f"{loc}_acc_{axis}" for loc in loc_list for axis in ["x", "y", "z"]]
        if args.num_sensor_channels != 3
        else ["acc_x", "acc_y", "acc_z"]
    )

    full_df = pd.DataFrame()
    for user in range(1, 16):
        for activity in activity_names:
            data = user_data[user].get(act_to_idx[activity])
            if data is None:
                continue
            cur = pd.DataFrame(data, columns=columns)
            cur["user"] = user
            cur["label"] = act_to_idx[activity]
            full_df = pd.concat([full_df, cur], ignore_index=True)

    print(f"Length of df {len(full_df)}")
    return full_df, columns


def get_data_from_split(df, args, split, n_fold=0):
    # Let us partition by train, val and test splits
    train_data = df[df["user"].isin(split["train"])]
    val_data = df[df["user"].isin(split["val"])]
    test_data = df[df["user"].isin(split["test"])]
    print(
        "The shapes of the splits are: {}, {} and {}".format(
            train_data.shape, val_data.shape, test_data.shape
        )
    )

    print(
        "The unique classes in train are: {}".format(
            np.unique(train_data["label"], return_counts=True)
        )
    )
    print(
        "The unique classes in val are: {}".format(
            np.unique(val_data["label"], return_counts=True)
        )
    )
    print(
        "The unique classes in test are: {}".format(
            np.unique(test_data["label"], return_counts=True)
        )
    )
    sensors = [c for c in df if c.endswith("acc_x") or c.endswith("acc_y") or c.endswith("acc_z")]

    processed = {
        "train": {
            "data": train_data[sensors].values,
            "labels": train_data["label"].values,
        },
        "val": {"data": val_data[sensors].values, "labels": val_data["label"].values},
        "test": {
            "data": test_data[sensors].values,
            "labels": test_data["label"].values,
        },
        "fold": split,
    }

    # Sanity check on the sizes
    for phase in ["train", "val", "test"]:
        assert processed[phase]["data"].shape[0] == len(processed[phase]["labels"])

    for phase in ["train", "val", "test"]:
        print(
            "The phase is: {}. The data shape is: {}, {}".format(
                phase, processed[phase]["data"].shape, processed[phase]["labels"].shape
            )
        )

    # Creating logs by the date now. To make stuff easier
    folder = os.path.join("all_data", date.today().strftime("%b-%d-%Y"))
    print(folder)
    os.makedirs(folder, exist_ok=True)

    os.makedirs(os.path.join(folder, "unnormalized"), exist_ok=True)
    if args.n_fold_validation != 0:
        save_name = f"rwhar_{len(sensors)}_sr_{args.sampling_rate}_fold_{n_fold}"
    else:
        save_name = f"rwhar_{len(sensors)}_sr_{args.sampling_rate}"

    # Before normalization
    print("Means before normalization")
    print(np.mean(processed["train"]["data"], axis=0))

    # Saving the joblib file
    save_name += ".joblib"
    name = os.path.join(folder, "unnormalized", save_name)
    with open(name, "wb") as f:
        dump(processed, f)

    # # Performing normalization
    if args.scaler_path is not None:
        scaler = load(args.scaler_path)
    else:
        scaler = StandardScaler()
        scaler.fit(processed["train"]["data"])
        if n_fold == 0:
            dump(scaler, os.path.join(folder, "./rwhar_norm.pkl"))

    for phase in ["train", "val", "test"]:
        processed[phase]["data"] = scaler.transform(processed[phase]["data"])

    # After normalization
    print("Means after normalization")
    print(np.mean(processed["train"]["data"], axis=0))

    # Saving into a joblib file
    name = os.path.join(folder, save_name)
    with open(name, "wb") as f:
        dump(processed, f)

    print("Saved into a joblib file!")

    return


def prepare_data(args):
    df, columns = get_data(args)
    unique_subj = np.unique(df["user"].values)

    if args.n_fold_validation == 0:
        split = perform_train_val_test_split(unique_subj, seed=args.seed)
        get_data_from_split(df, columns, args, split, n_fold=0)
    else:
        k = args.n_fold_validation
        num_test_subj = int(np.ceil(len(unique_subj) / k))
        sanity = {"train": [], "val": [], "test": []}

        for i in tqdm(range(k)):
            if i == 0:
                split = perform_train_val_test_split(unique_subj, seed=args.seed)
                train_subj, val_subj, test_subj = split.values()
            else:
                remaining_test = list(set(unique_subj) - set(sanity["test"]))
                random.shuffle(remaining_test)
                test_subj = (
                    remaining_test if i == k - 1 else remaining_test[:num_test_subj]
                )
                train_val = list(set(unique_subj) - set(test_subj))
                train_subj, val_subj = train_test_split(
                    train_val, test_size=0.2, random_state=args.seed
                )
                split = {"train": train_subj, "val": val_subj, "test": test_subj}

            sanity["train"].extend(train_subj)
            sanity["val"].extend(val_subj)
            sanity["test"].extend(test_subj)

            print(i, split)
            get_data_from_split(df, args, split, n_fold=i)

        assert len(set(sanity["test"])) == 15, (
            "Each subject should appear once in test across folds"
        )


# ---------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    # unzip_all("data/all_data/realworld2016_dataset/")  # NOTE: run first if the raw data is still zipped; this script expects .csv files
    args = parse_arguments()
    print(args)
    print("Seed", args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    prepare_data(args)
    print("Data preparation complete!")
