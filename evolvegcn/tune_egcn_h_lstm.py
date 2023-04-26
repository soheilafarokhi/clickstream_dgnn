import os
import sys
import subprocess
from ruamel.yaml import YAML

yaml = YAML()
yaml.boolean_representation = ['False', 'True']

command = "/home/soha/.virtualenvs/clicktream/bin/python run_exp.py --config_file=./experiments/parameters_click_stream_egcn_h_lstm.yaml"
i = 1
process_states = []
path = "experiments/parameters_click_stream_egcn_h_lstm.yaml"
with open(path) as f:
    json_doc = yaml.load(f)

courses = ['BBB', 'CCC', 'DDD', 'EEE', 'FFF', 'GGG', 'AAA']
weeks = [5, 10, 15, 20]
learning_rates = [0.001, 0.005, 0.01, 0.05, 1]
gcn_layer_1_feats = [250, 300, 350]
gcn_layer_2_feats = [250, 300, 350]
gcn_lstm_l2_feats = [128]
gcn_lstm_l2_hidden_dims = [128]
gcn_lstm_l2_dropouts = [ 0.3]
gcn_lstm_l2_layers = [1, 2]
gcn_cls_feats = [256, 512]
gcn_cls_l2_feats = [128, 256]
gcn_cls_dropouts = [0]
gcn_cls_l2_dropouts = [0]




for course in courses:
    if course == 'AAA':
        gcn_layer_1_feats = [128]
        gcn_layer_2_feats = [128]
    for num_weeks in weeks:
        for gcn_layer_1_feat in gcn_layer_1_feats:
            for gcn_layer_2_feat in gcn_layer_2_feats:
                for gcn_lstm_l2_feat in gcn_lstm_l2_feats:
                    for gcn_lstm_l2_hidden_dim in gcn_lstm_l2_hidden_dims:
                        for gcn_lstm_l2_dropout in gcn_lstm_l2_dropouts:
                            for gcn_lstm_l2_layer in gcn_lstm_l2_layers:
                                for gcn_cls_feat in gcn_cls_feats:
                                    for gcn_cls_l2_feat in gcn_cls_l2_feats:
                                        for gcn_cls_dropout in gcn_cls_dropouts:
                                            for gcn_cls_l2_dropout in gcn_cls_l2_dropouts:
                                                for alpha in learning_rates:
                                                    json_doc['tuning'] = True
                                                    json_doc['click_stream_args']['course'] = course
                                                    json_doc['num_hist_steps'] = num_weeks
                                                    json_doc['learning_rate'] = alpha
                                                    json_doc['gcn_parameters']['layer_1_feats'] = gcn_layer_1_feat
                                                    json_doc['gcn_parameters']['layer_2_feats'] = gcn_layer_2_feat
                                                    json_doc['gcn_parameters']['lstm_l2_feats'] = gcn_lstm_l2_feat
                                                    json_doc['gcn_parameters']['lstm_l2_hidden_dim'] = gcn_lstm_l2_hidden_dim
                                                    json_doc['gcn_parameters']['lstm_dropout'] = gcn_lstm_l2_dropout
                                                    json_doc['gcn_parameters']['lstm_l2_layers'] = gcn_lstm_l2_layer
                                                    json_doc['gcn_parameters']['cls_feats'] = gcn_cls_feat
                                                    json_doc['gcn_parameters']['cls_l2_feats'] = gcn_cls_l2_feat
                                                    json_doc['gcn_parameters']['cls_l1_dropout'] = gcn_cls_dropout
                                                    json_doc['gcn_parameters']['cls_l2_dropout'] = gcn_cls_l2_dropout

                                                    with open(path, "w") as f:
                                                        yaml.dump(json_doc, f)

                                                    i += 1
                                                    args = command.split(' ')
                                                    p = subprocess.Popen(args, cwd=r'/home/soha/workspace/py/clickstream/evolvegcn')
                                                    process_states.append(p)
                                                    # if len(process_states) == 15:
                                                    for j in range(len(process_states)):
                                                        process_states[j].wait()
                                                        print("process {} finished".format(i))
                                                    process_states = []
