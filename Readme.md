# Dynamic Graph Neural Networks for Student Performance Prediction

### Author: Soheila Farokhi & Arash Azizian Foumani

The initial dataset is located in the `data/` folder.

Please note that the course names in the dataset are different from their categorical title mentioned in the report. Refer to the following dictionary to find equivalen course names:

{ SS1 : AAA , SS2: BBB, SS3: GGG, ST1: DDD, ST2: CCC, ST3: EEE, ST4: FFF}

Baseline and graph dataset for our experiments are already available in the folder `saved_data/`.

The code for creating the baseline and graph datasets are in files `prepare_baseline_dataset.py` and 
`prepare_graph_dataset.py`, respectively.

To apply different traditional machine learning models including random forest on the baseline dataset, run the file `traditional_ml.py`.

To apply LSTM model on the baseline dataset with default hyper-parameters, run the `lstm_model.py` file. Alternatively, you can run the file `tune_lstm.py` with desired set of hyper-parameters.

To apply GCN+LSTM, EvolveGCN-H or EvolveGCN-H + LSTM model on the graph dataset, please refer to `Readme.md` file in evolvegcn directory. 