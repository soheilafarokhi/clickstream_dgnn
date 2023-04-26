import pandas as pd
import matplotlib.pyplot as plt
# rf_results = [0.69,0.73,0.78,0.81]
# lstm_results = [0.57,0.68,0.70,0.73]
# gcn_lstm_results = [0.68,0.61,0.48,0.72]
# egcn_results = [0.55,0.17,0.63,0.67]
# egcn_lstm_results = [0.60,0.60,0.71,0.70]
# course = 'SS1'

# rf_results = [0.69,0.74,0.78,0.81]
# lstm_results = [0.61,0.72,0.74,0.80]
# gcn_lstm_results = [0.59,0.72,0.75,0.80]
# egcn_results = [0.32,0.23,0.59,0.41]
# egcn_lstm_results = [0.61,0.70,0.76,0.79]
# course = 'SS2'

# rf_results = [0.66,0.71,0.75,0.78]
# lstm_results = [0.62,0.73,0.74,0.76]
# gcn_lstm_results = [0.48,0.70,0.71,0.71]
# egcn_results = [0.40,0.32,0.71,0.67]
# egcn_lstm_results = [0.54,0.70,0.69,0.77]
# course = 'ST1'

# rf_results = [0.70,0.77,0.77,0.80]
# lstm_results = [0.68,0.76,0.74,0.79]
# gcn_lstm_results = [0.68,0.74,0.74,0.76]
# egcn_results = [0.62,0.47,0.49,0.70]
# egcn_lstm_results = [0.68,0.73,0.75,0.74]
# course = 'ST2'

# rf_results = [0.75,0.83,0.84,0.89]
# lstm_results = [0.72,0.79,0.80,0.87]
# gcn_lstm_results = [0.74,0.77,0.78,0.79]
# egcn_results = [0.18,0.19,0.19,0.73]
# egcn_lstm_results = [0.74,0.79,0.80,0.80]
# course = 'ST3'


# rf_results = [0.74,0.80,0.85,0.88]
# lstm_results = [0.73,0.78,0.84,0.89]
# gcn_lstm_results = [0.73,0.77,0.83,0.89]
# egcn_results = [0.28,0.28,0.28,0.28]
# egcn_lstm_results = [0.73,0.79,0.84,0.89]
# course = 'ST4'
#
rf_results = [0.53,0.69,0.77,0.82]
lstm_results = [0.59,0.62,0.74,0.80]
gcn_lstm_results = [0.41,0.19,0.42,0.44]
egcn_results = [0.21,0.19,0.20,0.19]
egcn_lstm_results = [0.26,0.19,0.19,0.19]
course = 'SS3'

weeks = ['5','10','15','20']

df = pd.DataFrame({'Weeks':weeks, 'Random Forest':rf_results, 'LSTM':lstm_results, 'GCN+LSTM':gcn_lstm_results,
                   'EvolveGCN-H':egcn_results, 'EvolveGCN-H+LSTM':egcn_lstm_results})

df.plot(x='Weeks', y=['Random Forest', 'LSTM', 'GCN+LSTM',
                   'EvolveGCN-H', 'EvolveGCN-H+LSTM'], kind='line', figsize=(5,5))
plt.title(f'Course {course}')
plt.ylabel('Weighted F1')
plt.legend(loc='lower right')
plt.show()