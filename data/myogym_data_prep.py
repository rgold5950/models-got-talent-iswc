import argparse
from datetime import date

import numpy as np
import os
import pandas as pd
import random
import scipy.io as sio
from joblib import dump
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

np.random.seed(42)
random.seed(42)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Parameters for '
                                                 'preparing Myogym dataset')
    parser.add_argument('--dataset_loc', type=str,
                        default='data/raw/myogym/MyoGym.mat',
                        help='Location of the raw sensory data (see docs/DATASETS.md)')
    parser.add_argument('--sampling_rate', type=int, default=50,
                        help='Sampling rate for the data. Is used to '
                             'downsample to the required rate')
    parser.add_argument('--perform_normalization', action='store_true',
                        help='Enable mean-variance normalization')
    parser.add_argument('--num_sensor_channels', type=int, default=3,
                        help='Number of sensor channels for used in the data '
                             'preparation')
    parser.add_argument('--n_fold_validation', type=int, default=0,
                        help='To extract data with n-folds instead of random '
                             '20% test set. Default is 0, which creates the '
                             'normal 80-20 split.')

    args = parser.parse_args()

    return args


def map_activity_to_id():
    # List of activities being studied. Note that we *dont* use lying down class
    activity_list = [
        'NULL', 'Seated Cable Rows', 'One-Arm Dumbbell Row',
        'Wide-Grip Pulldown Behind The Neck', 'Bent Over Barbell Row',
        'Reverse Grip Bent-Over Row', 'Wide-Grip Front Pulldown', 'Bench Press',
        'Incline Dumbbell Flyes', 'Incline Dumbbell Press', 'Dumbbell Flyes',
        'Pushups', 'Leverage Chest Press', 'Close-Grip Barbell Bench Press',
        'Bar Skullcrusher', 'Triceps Pushdown', 'Bench Dip / Dip',
        'Overhead Triceps Extension', 'Tricep Dumbbell Kickback', 'Spider Curl',
        'Dumbbell Alternate Bicep Curl', 'Incline Hammer Curl',
        'Concentration Curl', 'Cable Curl', 'Hammer Curl',
        'Upright Barbell Row', 'Side Lateral Raise', 'Front Dumbbell Raise',
        'Seated Dumbbell Shoulder Press', 'Car Drivers',
        'Lying Rear Delt Raise']

    activity_id = {
        'Seated Cable Rows': 1,
        'One-Arm Dumbbell Row': 2,
        'Wide-Grip Pulldown Behind The Neck': 3,
        'Bent Over Barbell Row': 4,
        'Reverse Grip Bent-Over Row': 5,
        'Wide-Grip Front Pulldown': 6,
        'Bench Press': 7,
        'Incline Dumbbell Flyes': 8,
        'Incline Dumbbell Press': 9,
        'Dumbbell Flyes': 10,
        'Pushups': 11,
        'Leverage Chest Press': 12,
        'Close-Grip Barbell Bench Press': 13,
        'Bar Skullcrusher': 14,
        'Triceps Pushdown': 15,
        'Bench Dip / Dip': 16,
        'Overhead Triceps Extension': 17,
        'Tricep Dumbbell Kickback': 18,
        'Spider Curl': 19,
        'Dumbbell Alternate Bicep Curl': 20,
        'Incline Hammer Curl': 21,
        'Concentration Curl': 22,
        'Cable Curl': 23,
        'Hammer Curl': 24,
        'Upright Barbell Row': 25,
        'Side Lateral Raise': 26,
        'Front Dumbbell Raise': 27,
        'Seated Dumbbell Shoulder Press': 28,
        'Car Drivers': 29,
        'Lying Rear Delt Raise': 30,
        'NULL': 0}

    return activity_id, activity_list


def perform_train_val_test_split(unique_subj, test_size=0.2, val_size=0.2):
    # Doing the train-test split
    train_val_subj, test_subj = train_test_split(unique_subj,
                                                 test_size=test_size,
                                                 random_state=42)
    print('The train and validation subjects are: {}'.format(train_val_subj))
    print('The test subjects are: {}'.format(test_subj))

    # Splitting further into train and validation subjects
    train_subj, val_subj = train_test_split(train_val_subj, test_size=val_size,
                                            random_state=42)

    subjects = {'train': train_subj, 'val': val_subj, 'test': test_subj}

    pd.to_pickle(subjects, 'subjects.pkl')

    return subjects


def get_data(args):
    # Loading the mat file
    dataset = loadmat(args.dataset_loc)
    raw_data = dataset['raw_data']
    labels_all = dataset['raw_data_labels']
    labels = labels_all[:, 0]
    participant_id = labels_all[:, 1]

    # Downsampling upfront
    divide_by = int(50 / args.sampling_rate)
    index = np.arange(0, len(raw_data), divide_by)

    # Taking the data at those indices
    sensor_data = raw_data[index]
    labels = labels[index]
    participant_id = participant_id[index]

    # Changing from 99 to NULL
    labels[labels == 99] = 0

    # Adding to the data frame
    df = pd.DataFrame()
    df['user'] = participant_id
    df['acc_x'] = sensor_data[:, 10]
    df['acc_y'] = sensor_data[:, 11]
    df['acc_z'] = sensor_data[:, 12]
    df['gyro_x'] = sensor_data[:, 14]
    df['gyro_y'] = sensor_data[:, 15]
    df['gyro_z'] = sensor_data[:, 16]
    df['gt'] = labels

    print('Done collecting!')
    return df


def get_data_from_split(df, args, split, n_fold=0):
    # Let us partition by train, val and test splits
    train_data = df[df['user'].isin(split['train'])]
    val_data = df[df['user'].isin(split['val'])]
    test_data = df[df['user'].isin(split['test'])]
    print('The shapes of the splits are: {}, {} and {}'.
          format(train_data.shape, val_data.shape, test_data.shape))

    print('The unique classes in train are: {}'
          .format(np.unique(train_data['gt'])))
    print('The unique classes in val are: {}'
          .format(np.unique(val_data['gt'])))
    print('The unique classes in test are: {}'
          .format(np.unique(test_data['gt'])))

    if args.num_sensor_channels == 3:
        sensors = ['acc_x', 'acc_y', 'acc_z']
    elif args.num_sensor_channels == 6:
        sensors = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']

    processed = {'train': {'data': train_data[sensors].values,
                           'labels': train_data['gt'].values},
                 'val': {'data': val_data[sensors].values,
                         'labels': val_data['gt'].values},
                 'test': {'data': test_data[sensors].values,
                          'labels': test_data['gt'].values},
                 'fold': split
                 }

    # Sanity check on the sizes
    for phase in ['train', 'val', 'test']:
        assert processed[phase]['data'].shape[0] == \
               len(processed[phase]['labels'])

    for phase in ['train', 'val', 'test']:
        print('The phase is: {}. The data shape is: {}, {}'
              .format(phase, processed[phase]['data'].shape,
                      processed[phase]['labels'].shape))

    # Before normalization
    print('Means before normalization')
    print(np.mean(processed['train']['data'], axis=0))

    # Creating logs by the date now. To make stuff easier
    folder = os.path.join('all_data', date.today().strftime(
        "%b-%d-%Y"))
    os.makedirs(folder, exist_ok=True)

    os.makedirs(os.path.join(folder, 'unnormalized'), exist_ok=True)
    args.n_fold = n_fold
    if args.n_fold_validation != 0:
        save_name = 'myogym_{0.num_sensor_channels}_sr_' \
                    '{0.sampling_rate}_fold_{0.n_fold}.joblib'.format(args)
    else:
        save_name = 'myogym_{0.num_sensor_channels}_sr_{0.sampling_rate}' \
                    '.joblib'.format(args)

    name = os.path.join(folder, 'unnormalized', save_name)
    with open(name, 'wb') as f:
        dump(processed, f)

    # Performing normalization
    scaler = StandardScaler()
    scaler.fit(processed['train']['data'])
    for phase in ['train', 'val', 'test']:
        processed[phase]['data'] = \
            scaler.transform(processed[phase]['data'])

    # After normalization
    print('Means after normalization')
    print(np.mean(processed['train']['data'], axis=0))

    # Saving into a joblib file
    name = os.path.join(folder, save_name)
    with open(name, 'wb') as f:
        dump(processed, f)
    print('Saved into a joblib file!')

    return


def prepare_data(args):
    # Loading in all the data first
    df = get_data(args=args)

    # Getting the unique subject IDs for splitting
    unique_subj = np.unique(df['user'].values)
    print('The unique subjects are: {}'.format(unique_subj))

    # Performing the train-val-test split
    if args.n_fold_validation == 0:
        split = perform_train_val_test_split(unique_subj)
        get_data_from_split(df, args, split)
    else:
        n_fold_validation = 5
        num_test_subj = int((1.0 / n_fold_validation) * len(unique_subj))
        print('The number of validation and test subjects: '
              '{}'.format(num_test_subj))

        sanity = {'train': [], 'val': [], 'test': []}

        for i in tqdm(range(n_fold_validation)):
            # First fold is same as random 80:20 split
            if i == 0:
                split = perform_train_val_test_split(unique_subj)
                train_subj = split['train']
                val_subj = split['val']
                test_subj = split['test']
            else:
                remaining_test = list(set(unique_subj) - set(sanity['test']))

                # Going to shuffle it in place and pick the first num_test_subj
                np.random.shuffle(remaining_test)

                if i != n_fold_validation - 1:
                    test_subj = remaining_test[:num_test_subj]
                else:
                    test_subj = remaining_test

                # Remaining participants for train+val
                train_val = list(set(unique_subj) - set(test_subj))

                # Splitting that 80:20
                train_subj, val_subj = train_test_split(train_val,
                                                        test_size=0.2,
                                                        random_state=42)

            # Sanity check to make sure all subjects were in test/val
            # once only
            sanity['train'].extend(train_subj)
            sanity['val'].extend(val_subj)
            sanity['test'].extend(test_subj)

            assert len(test_subj) == 2

            subjects = {'train': train_subj, 'val': val_subj, 'test': test_subj}
            print(i, subjects)

            # Saving the split data
            get_data_from_split(df, args, split=subjects, n_fold=i)

        # For test split, there have to be each participant only once
        print(sanity)
        assert len(sanity['test']) == 10
        assert sorted(sanity['test']) == sorted(list(unique_subj))


        v, c = np.unique(sanity['test'], return_counts=True)
        assert np.sum(c == 1) == 10

    return

if __name__ == '__main__':
    args = parse_arguments()
    print(args)

    prepare_data(args)
    print('Data preparation complete!')