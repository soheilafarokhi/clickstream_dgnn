import pandas as pd
#course = 'GGG'
# df = pd.read_csv(f'./evolvegcn/egcn_data/train_phase/{course}_2/node_labels_train.csv')
# df_dev = pd.read_csv(f'./evolvegcn/egcn_data/train_phase/{course}_2/node_labels_dev.csv')
# df_test = pd.read_csv(f'./evolvegcn/egcn_data/train_phase/{course}_2/node_labels_test.csv')
# print(len(df.loc[df['1'] == 1]))
# print(len(df.loc[df['1'] == 0]))
# print(len(df))
#
# print(len(df_dev.loc[df_dev['1'] == 1]))
# print(len(df_dev.loc[df_dev['1'] == 0]))
# print(len(df_dev))
#
# print(len(df_test.loc[df_test['1'] == 1]))
# print(len(df_test.loc[df_test['1'] == 0]))
# print(len(df_test))

import re
import numpy as np
courses = ['EEE', 'FFF', 'GGG']#'AAA', 'BBB', 'CCC', 'DDD']
weeks = [5, 10,15,20]
dictionaries = []
num_classes = 2
model = 'gcn_lstm'
for course in courses:
    print(course)
    for num_week in weeks:
        print(num_week)
        with open(f'./log/tuning/{model}/{course}_{num_classes}/log_{num_week}_weeks.txt', "r") as f:
            string = f.read()
        f1_values = re.findall(r"Best valid measure:(\d+\.\d+)", string)
        f1_values = [float(x) for x in f1_values]
        # configs = re.findall("{(.+?)}", string)
        # timestamps = re.findall(r"F1-score: (\d+\.\d+)", string)
        # timestamps = [str(x) for x in timestamps]
        # max_f1_index = np.argmax(f1_values)
        max_f1 = np.max(f1_values)
        print(max_f1)
        for item in string.split("\n"):
            # print(item)
            if f"Best valid measure:{max_f1}" in item:
                timestamp = item.strip().split(' ')[0]
                print(f'model timestamp is: {timestamp}')
