import os

import pandas as pd
import random
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split

print(os.getcwd())

df = pd.read_csv('./data/clickstream_dataset_updated.csv')
df.columns = range(df.columns.size)
df.rename({0: 'timestamp', 1: 'course_id', 2: 'user_id'}, axis=1, inplace=True)

num_classes = 2
df.loc[df[640] < 2, 640] = 0
df.loc[df[640] > 0, 640] = 1

courses = pd.unique(df['course_id'])

random.seed = 42

x = np.array(range(0, 20), dtype=int)
b = np.zeros((x.size, x.max() + 1), dtype=int)
b[np.arange(x.size), x] = 1
b_df = pd.DataFrame(b)
empty_user_features = pd.DataFrame(np.zeros((20, 37), dtype=int))
behaviour_features = pd.concat([empty_user_features, b_df], axis=1, ignore_index=True)
behaviour_features = behaviour_features.reset_index()
behaviour_features.columns = pd.RangeIndex(0, len(behaviour_features.columns))

bl = np.zeros((20, 1), dtype=int) + num_classes
behaviour_labels = pd.DataFrame(bl)
behaviour_labels = behaviour_labels.reset_index()
behaviour_labels.columns = pd.RangeIndex(0, len(behaviour_labels.columns))


def store_final_data(semester_df, phase, course, version):
    label_column = semester_df.iloc[:, -1]
    user_index = 20
    user_dict = {}
    for index, row in semester_df.iterrows():
        if row['user_id'] in user_dict:
            semester_df.loc[index, 'user_id'] = user_dict[row['user_id']]
        else:
            semester_df.loc[index, 'user_id'] = user_index
            user_dict[row['user_id']] = user_index
            user_index = user_index + 1
        if (row[3:603] == 0).all():
            n = random.randint(3, 602)
            semester_df.loc[index, n] = 0.001

    user_column = semester_df['user_id']
    # print(user_column.shape)
    user_features = semester_df.iloc[:, 603:640]
    if version == 'train':
        user_features = pd.DataFrame(scaler.fit_transform(user_features))
    elif (version == 'dev') or (version == 'test'):
        user_features = pd.DataFrame(scaler.transform(user_features))

    # display(user_features)
    empty_behaviour_features = pd.DataFrame(np.zeros((len(semester_df), 20), dtype=int))
    # print(empty_behaviour_features.shape)
    node_features = pd.concat([user_column, user_features], axis=1)
    # display(node_features)
    node_features = pd.concat([node_features, empty_behaviour_features], axis=1, ignore_index=True)
    # display(node_features)
    node_features = pd.concat([behaviour_features, node_features], ignore_index=True)
    # print(node_features.shape)

    if not os.path.exists(f"./saved_data/graph/{phase}/{course}_{num_classes}"):
        os.makedirs(f"./saved_data/graph/{phase}/{course}_{num_classes}")
    node_features.to_csv(f'./saved_data/graph/{phase}/{course}_{num_classes}/node_feat_{version}.csv',
                         index=False)

    node_labels = pd.concat([user_column, label_column], axis=1, ignore_index=True)
    node_labels = pd.concat([behaviour_labels, node_labels], ignore_index=True)

    node_labels.to_csv(f'./saved_data/graph/{phase}/{course}_{num_classes}/node_labels_{version}.csv',
                       index=False)
    for i in range(3, 603, 20):
        temp_df = semester_df.iloc[:, i:i + 20]

        temp_df = pd.concat([user_column, temp_df], axis=1)
        temp_df.columns = ['user_id' if i == 0 else 'b' + str(i) for i, x in enumerate(temp_df.columns)]
        stacked = temp_df.set_index(['user_id']).stack()
        stacked = stacked.reset_index(name='clicks')
        stacked = stacked.rename({'level_1': 'behavioural_data'}, axis=1)
        stacked = stacked.loc[stacked['clicks'] != 0]
        stacked['behavioural_data'] = stacked['behavioural_data'].str[1:]
        stacked['behavioural_data'] = stacked['behavioural_data'].astype(int)
        stacked['behavioural_data'] = stacked['behavioural_data'] - 1

        stacked.to_csv(
            f'./saved_data/graph/{phase}/{course}_{num_classes}/el_{version}_week_{int((i - 3) / 20)}.csv',
            index=False)


train_dfs = []
dev_dfs = []
test_dfs = []
phases = ['train_phase', 'test_phase']
for phase in phases:
    for course in courses:
        scaler = StandardScaler()
        course_df = df.loc[df['course_id'] == course]
        course_df = course_df.reset_index(drop=True)
        semesters = pd.unique(course_df['timestamp'])
        semesters = sorted(semesters, key=str)
        print(course)
        print(semesters)
        next_to_last_semester = ''
        if (course != 'AAA') and (course != 'CCC'):
            next_to_last_semester = semesters[-2]
        last_semester = semesters[-1]

        for j in range(3):
            if (j == 1) and ((phase == 'test_phase') or (course in ['AAA', 'CCC'])):
                continue

            if j == 0:
                version = 'train'
                if phase == 'train_phase':
                    other_semesters = [last_semester, next_to_last_semester]
                else:
                    other_semesters = [last_semester]
                semester_df = course_df.loc[~course_df['timestamp'].isin(other_semesters)]
                semester_df = semester_df.reset_index(drop=True)

                if course in ['AAA', 'CCC'] and (phase != 'test_phase'):
                    X = semester_df.iloc[:, :-1]
                    y = semester_df.iloc[:, -1:]
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42,
                                                                        stratify=y)

                    semester_df = pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1,
                                            ignore_index=True)
                    semester_df = semester_df.reset_index(drop=True)
                    dev_df = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1,
                                       ignore_index=True)
                    dev_df = dev_df.reset_index(drop=True)
                    semester_df.rename({0: 'timestamp', 1: 'course_id', 2: 'user_id'}, axis=1, inplace=True)
                    dev_df.rename({0: 'timestamp', 1: 'course_id', 2: 'user_id'}, axis=1, inplace=True)
                    dev_dfs.append(dev_df)
                train_dfs.append(semester_df)
            elif j == 1:
                version = 'dev'
                semester_df = course_df.loc[course_df['timestamp'] == next_to_last_semester]
                semester_df = semester_df.reset_index(drop=True)
                dev_dfs.append(semester_df)
            else:
                version = 'test'
                semester_df = course_df.loc[course_df['timestamp'] == last_semester]
                semester_df = semester_df.reset_index(drop=True)
                test_dfs.append(semester_df)

            store_final_data(semester_df, phase, course, version)
            if course in ['AAA', 'CCC'] and (version == 'train') and (phase != 'test_phase'):
                store_final_data(dev_df, phase, course, 'dev')

    course = 'all'
    scaler = StandardScaler()
    for j in range(3):
        if (j == 1) and (phase == 'test_phase'):
            continue
        if j == 0:
            version = 'train'
            semester_df = pd.concat(train_dfs, ignore_index=True)
        elif j == 1:
            version = 'dev'
            semester_df = pd.concat(dev_dfs, ignore_index=True)
        else:
            version = 'test'
            semester_df = pd.concat(test_dfs, ignore_index=True)

        store_final_data(semester_df, phase, course, version)
