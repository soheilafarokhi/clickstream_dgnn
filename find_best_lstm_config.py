import re
import numpy as np
courses = ['AAA',  'CCC']
weeks = [5,10,15,20]
dictionaries = []
for course in courses:
    print(course)
    for num_week in weeks:
        print(num_week)
        with open(f'./final_results/{course}_{num_week}_weeks_lstm_validation_f1.txt', "r") as f:
            string = f.read()
        f1_values = re.findall(r"F1-score: (\d+\.\d+)", string)
        f1_values = [float(x) for x in f1_values]
        configs = re.findall("{(.+?)}", string)
        timestamps = re.findall(r"F1-score: (\d+\.\d+)", string)
        timestamps = [str(x) for x in timestamps]

        max_f1_index = np.argmax(f1_values)

        with open(f'./final_results/{course}_{num_week}_weeks_lstm_grid_search_results.txt', "r") as f:
            grid_string = f.readlines()

        count = 0
        # Strips the newline character
        flag = False
        valid_reports = []
        index = 0
        for line in grid_string:
            if 'Validation results:' in str(line):
                flag = True
            elif flag == True:
                if count == 0:
                    valid_reports.append(str(line.strip()))
                else:
                    valid_reports[index] = valid_reports[index] + str(line)
                count += 1

            if count == 8:
                flag = False
                count = 0
                index += 1

        count = 0
        # Strips the newline character
        flag = False
        test_reports = []
        index = 0
        for line in grid_string:
            if 'Test results:' in str(line):
                flag = True
            elif flag == True:
                if count == 0:
                    test_reports.append(str(line.strip()))
                else:
                    test_reports[index] = test_reports[index] + str(line)
                count += 1

            if count == 8:
                flag = False
                count = 0
                index += 1
        Dict = eval('{' + configs[max_f1_index] + '}')
        dictionaries.append(Dict)
        print(np.max(f1_values))
        print('validation')
        print(valid_reports[max_f1_index])
        print('test')
        print(test_reports[max_f1_index])
print(dictionaries)
