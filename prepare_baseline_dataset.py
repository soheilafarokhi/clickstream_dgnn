import pandas as pd
import os
from sklearn.model_selection import train_test_split

df = pd.read_csv('./data/clickstream_dataset_updated.csv')
df.columns = range(df.columns.size)
df.rename({0: 'timestamp', 1: 'course_id', 2: 'user_id'}, axis=1, inplace=True)

num_classes = 2
df.loc[df[640] < 2, 640] = 0
df.loc[df[640] > 0, 640] = 1

courses = pd.unique(df['course_id'])

phases = ['train_phase', 'test_phase']
for phase in phases:
    for course in courses:
        # scaler = StandardScaler()
        course_df = df.loc[df['course_id'] == course]
        course_df = course_df.reset_index(drop=True)
        student_count = pd.unique(course_df['user_id'])
        print(course)
        print(f'Number of unique students: {student_count}')
        semesters = pd.unique(course_df['timestamp'])
        semesters = sorted(semesters, key=str)

        next_to_last_semester = ''
        if (course != 'AAA') and (course != 'CCC'):
            next_to_last_semester = semesters[-2]
        last_semester = semesters[-1]

        for j in range(3):
            if (j == 1) and ((phase == 'test_phase') or (course in ['AAA','CCC'])):
                continue
            if j == 0:
                version = 'train'
                if phase == 'train_phase':
                    other_semesters = [last_semester, next_to_last_semester]
                else:
                    other_semesters = [last_semester]
                semester_df = course_df.loc[~course_df['timestamp'].isin(other_semesters)]
                if course in ['AAA', 'CCC'] and (phase != 'test_phase'):

                    X = semester_df.iloc[:, :-1]
                    y = semester_df.iloc[:, -1:]
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

                    semester_df = pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1, ignore_index=True)
                    dev_df = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1, ignore_index=True)
                    dev_df = dev_df.iloc[:, 3:]
                    # semester_df = semester_df.drop(range(603, 640), axis=1)
                    dev_df = dev_df.reset_index(drop=True)
                    dev_df.columns = range(dev_df.columns.size)

                    dev_df.to_csv(f'./saved_data/baseline/{phase}/{course}_{num_classes}/dev_dataset.csv',
                                       index=False)
            elif j == 1:
                version = 'dev'
                semester_df = course_df.loc[course_df['timestamp'] == next_to_last_semester]
            else:
                version = 'test'
                semester_df = course_df.loc[course_df['timestamp'] == last_semester]

            semester_df = semester_df.iloc[:, 3:]
            # semester_df = semester_df.drop(range(603, 640), axis=1)
            semester_df = semester_df.reset_index(drop=True)
            semester_df.columns = range(semester_df.columns.size)
            if not os.path.exists(f"./saved_data/baseline/{phase}/{course}_{num_classes}"):
                os.makedirs(f"./saved_data/baseline/{phase}/{course}_{num_classes}")
            semester_df.to_csv(f'./saved_data/baseline/{phase}/{course}_{num_classes}/{version}_dataset.csv',
                               index=False)