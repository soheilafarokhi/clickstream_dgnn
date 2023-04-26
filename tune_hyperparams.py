import os
import sys
import subprocess

# args =config.args
# command_template = "CUDA_VISBLE_DEVICES={}, python main.py --alpha {} --classes {}"
command_template = "/home/soha/.virtualenvs/clicktream/bin/python best_lstm.py --course {} --num_weeks {} --learning_rate {} --dropout {} --hidden_dim {} --num_layers {} --epochs {} --fcn_dim {}"
i = 1
process_states = []

# for c1 in range(0,10):
#     for c2 in range(0,10):
#         if c1 >= c2:
#             continue
courses = ['AAA', 'CCC']
weeks = [5, 10, 15, 20]
epochs = [100, 200]
dropouts = [0.2, 0.3, 0.5]
num_layers = [2, 3]
hidden_dims = [64, 128]
learning_rates = [0.001, 0.01, 0.1, 0.5, 1]
fcn_dims = [16, 32, 64]
for course in courses:
    for num_weeks in weeks:
        for epoch in epochs:
            for num_layer in num_layers:
                for hidden_dim in hidden_dims:
                    for dropout in dropouts:
                        for fcn_dim in fcn_dims:
                            for alpha in learning_rates:
                                command = command_template.format(course, num_weeks, alpha, dropout, hidden_dim,
                                                                  num_layer, epoch, fcn_dim)
                                print(i, command)
                                i += 1
                                args = command.split(' ')
                                p = subprocess.Popen(args, cwd=r'/home/soha/workspace/py/clickstream')
                                process_states.append(p)
                                # if len(process_states) == 15:
                                for j in range(len(process_states)):
                                    process_states[j].wait()
                                    print("process {} finished".format(i))
                                process_states = []
