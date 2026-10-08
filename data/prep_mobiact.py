import argparse
import os
import pickle
from datetime import date

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

np.random.seed(42)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Parameters for preparing Mobiactv2")
    parser.add_argument(
        "--dataset_loc",
        type=str,
        default="data/raw/mobiactv2/MobiAct_Dataset_v2.0/Annotated_Data",
        help="Location of the raw sensory data (see docs/DATASETS.md)",
    )
    parser.add_argument(
        "--sampling_rate",
        type=int,
        default=50,  # NOTE: this should almost always be run at 50hz
        help="Sampling rate for the data. Is used to downsample to the required rate",
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

    args = parser.parse_args()

    return args


def map_activity_to_id(args):
    # List of activities being studied. Note that we *dont* use lying down class
    activity_list = [
        "STD",
        "WAL",
        "JOG",
        "JUM",
        "STU",
        "STN",
        "SCH",
        "SIT",
        "CHU",
        "CSI",
        "CSO",
    ]

    if args.null_class == "False":
        activity_id = {
            "STD": 0,
            "WAL": 1,
            "JOG": 2,
            "JUM": 3,
            "STU": 4,
            "STN": 5,
            "SCH": 6,
            "SIT": 7,
            "CHU": 8,
            "CSI": 9,
            "CSO": 10,
        }
    else:
        activity_id = {
            "STD": 0,
            "WAL": 1,
            "JOG": 2,
            "JUM": 3,
            "STU": 4,
            "STN": 5,
            "SCH": 7,
            "SIT": 6,
            "CHU": 7,
            "CSI": 7,
            "CSO": 7,
        }

    return activity_id, activity_list


def perform_train_val_test_split(unique_subj, test_size=0.2, val_size=0.2):
    # Doing the train-test split
    train_val_subj, test_subj = train_test_split(
        unique_subj, test_size=test_size, random_state=42
    )
    print("The train and validation subjects are: {}".format(train_val_subj))
    print("The test subjects are: {}".format(test_subj))

    # Splitting further into train and validation subjects
    train_subj, val_subj = train_test_split(
        train_val_subj, test_size=val_size, random_state=42
    )

    subjects = {"train": train_subj, "val": val_subj, "test": test_subj}

    folder = os.path.join("all_data/mobiactv2/", date.today().strftime("%b-%d-%Y"))
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "subjects.pkl"), "wb") as f:
        pickle.dump(subjects, f, pickle.HIGHEST_PROTOCOL)

    return subjects


def get_data(args):
    # Getting the activity labels
    activity_id, activity_list = map_activity_to_id(args=args)

    shapes = []
    flag = 0
    cols = []

    for i in tqdm(range(0, len(activity_list))):
        folder = os.path.join(args.dataset_loc, activity_list[i])
        files = os.listdir(folder)
        # print(folder, len(files))

        # Looping over all the files to concatenate the data present inside
        for j in range(0, len(files)):
            # print(files[j])
            participant_id = files[j].split("_")[1]

            csv_file = os.path.join(folder, files[j])
            data = pd.read_csv(csv_file)

            # Remove the lying label
            data = data[data["label"] != "LYI"]

            # Replace the labels with numerical ones
            data["label"] = data["label"].map(activity_id)

            # Sampling it down to the required sampling rate ~33.33Hz
            index = np.arange(0, len(data), (200 / args.sampling_rate)).astype(np.int32)
            sampled = data.iloc[index, :]
            shapes.append(sampled.shape[0])

            # Participants
            participant_list = np.ones(len(sampled)) * int(participant_id)
            # sampled['user'] = participant_list
            sampled = sampled.assign(user=participant_list)
            cols = sampled.columns

            if flag == 0:
                har_data = sampled.values
                flag = 1
            else:
                har_data = np.vstack((har_data, sampled.values))

    # Putting it back into a dataframe
    df_cols = {}
    for i in range(len(cols)):
        df_cols[cols[i]] = []

    # Putting into dataframe
    df = pd.DataFrame(df_cols)
    df["timestamp"] = har_data[:, 0]
    df["rel_time"] = har_data[:, 1]
    df["acc_x"] = har_data[:, 2]
    df["acc_y"] = har_data[:, 3]
    df["acc_z"] = har_data[:, 4]
    df["gyro_x"] = har_data[:, 5]
    df["gyro_y"] = har_data[:, 6]
    df["gyro_z"] = har_data[:, 7]
    df["azimuth"] = har_data[:, 8]
    df["pitch"] = har_data[:, 9]
    df["roll"] = har_data[:, 10]
    df["label"] = har_data[:, 11]
    df["user"] = har_data[:, 12]

    print("Done collecting!")
    return df


def get_data_from_split(df, split, args, n_fold=0):
    # Let us partition by train, val and test splits
    train_data = df[df["user"].isin(split["train"])]
    val_data = df[df["user"].isin(split["val"])]
    test_data = df[df["user"].isin(split["test"])]
    print(
        "The shapes of the splits are: {}, {} and {}".format(
            train_data.shape, val_data.shape, test_data.shape
        )
    )

    print("The unique classes in train are: {}".format(np.unique(train_data["label"])))
    print("The unique classes in val are: {}".format(np.unique(val_data["label"])))
    print("The unique classes in test are: {}".format(np.unique(test_data["label"])))

    if args.num_sensor_channels == 3:
        sensors = ["acc_x", "acc_y", "acc_z"]
    elif args.num_sensor_channels == 6:
        sensors = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]

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

    # Before normalization
    print("Means before normalization")
    print(np.mean(processed["train"]["data"], axis=0))

    # Creating logs by the date now. To make stuff easier
    folder = os.path.join("all_data", date.today().strftime("%b-%d-%Y"))
    os.makedirs(folder, exist_ok=True)

    os.makedirs(os.path.join(folder, "unnormalized"), exist_ok=True)
    args.n_fold = n_fold
    if args.n_fold_validation != 0:
        save_name = "mobiactv2_6_sr_{0.sampling_rate}_fold_{0.n_fold}".format(args)
    else:
        save_name = "mobiactv2_6_sr_{0.sampling_rate}".format(args)

    # If null class, add _null to the end
    if args.null_class == "True":
        save_name += "_null"

    # Saving the joblib file
    save_name += ".joblib"
    name = os.path.join(folder, "unnormalized", save_name)
    with open(name, "wb") as f:
        dump(processed, f)

    # Performing normalization
    scaler = StandardScaler()
    scaler.fit(processed["train"]["data"])
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
    # Reading in all the data
    df = get_data(args=args)

    # Getting the unique subject IDs for splitting
    unique_subj = np.unique(df["user"].values)
    print("The unique subjects are: {}".format(unique_subj))

    # Performing the train-val-test split
    if args.n_fold_validation == 0:
        split = perform_train_val_test_split(unique_subj)
        get_data_from_split(df, split, args)
    else:
        n_fold_validation = args.n_fold_validation
        num_test_subj = int(np.ceil((1.0 / n_fold_validation) * len(unique_subj)))
        print("The number of validation and test subjects: {}".format(num_test_subj))

        sanity = {"train": [], "val": [], "test": []}

        for i in range(n_fold_validation):
            # First fold is same as random 80:20 split
            if i == 0:
                split = perform_train_val_test_split(unique_subj)
                train_subj = split["train"]
                val_subj = split["val"]
                test_subj = split["test"]
            else:
                remaining_test = list(set(unique_subj) - set(sanity["test"]))

                # Going to shuffle it in place and pick the first num_test_subj
                np.random.shuffle(remaining_test)

                if i != n_fold_validation - 1:
                    test_subj = remaining_test[:num_test_subj]
                else:
                    test_subj = remaining_test

                # Remaining participants for train+val
                train_val = list(set(unique_subj) - set(test_subj))

                # Splitting that 80:20
                train_subj, val_subj = train_test_split(
                    train_val, test_size=0.2, random_state=42
                )

            # Sanity check to make sure all subjects were in test/val once only
            sanity["train"].extend(train_subj)
            sanity["val"].extend(val_subj)
            sanity["test"].extend(test_subj)

            subjects = {"train": train_subj, "val": val_subj, "test": test_subj}
            print(i, subjects)

            get_data_from_split(df, split=subjects, args=args, n_fold=i)

        # For test split, there have to be each participant only
        # once
        print(sanity)
        assert len(sanity["test"]) == 61

        v, c = np.unique(sanity["test"], return_counts=True)
        assert np.sum(c == 1) == 61

    return


if __name__ == "__main__":
    args = parse_arguments()
    print(args)

    prepare_data(args)
    print("Data preparation complete!")
