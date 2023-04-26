import time

from torch import nn
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import os
import glob
from torch.utils.data import DataLoader
import pandas as pd
import re
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import argparse
import json
from sklearn.metrics import f1_score

# INITIALIZE PARAMETERS
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', default=100, type=int, help='Number of epochs to train the model')
parser.add_argument('--hidden_dim', default=64, type=int, help='Number of dimensions of the dynamic embedding')
parser.add_argument('--learning_rate', default=0.001, type=float, help='Learning rate')
parser.add_argument('--dropout', default=0.3, type=float, help='Learning rate')
parser.add_argument('--weight_decay', default=0.0001, type=float, help='Learning rate')
parser.add_argument('--fcn_dim', default=32, type=int, help='Learning rate')
parser.add_argument('--num_layers', default=1, type=int, help='Learning rate')
parser.add_argument('--course', default='BBB', type=str, help='Learning rate')
parser.add_argument('--num_weeks', default=5, type=int, help='Learning rate')
parser.add_argument('--num_classes', default=2, type=int, help='Learning rate')


args = parser.parse_args()
configs = [{'epochs': 100,
  'hidden_dim': 64,
  'learning_rate': 0.001,
  'dropout': 0.2,
  'weight_decay': 0.0001,
  'fcn_dim': 64,
  'num_layers': 3,
  'course': 'BBB',
  'num_weeks': 5,
  'num_classes': 2},
 {'epochs': 100,
  'hidden_dim': 128,
  'learning_rate': 0.1,
  'dropout': 0.2,
  'weight_decay': 0.0001,
  'fcn_dim': 16,
  'num_layers': 3,
  'course': 'BBB',
  'num_weeks': 10,
  'num_classes': 2},
 {'epochs': 100,
  'hidden_dim': 128,
  'learning_rate': 0.001,
  'dropout': 0.2,
  'weight_decay': 0.0001,
  'fcn_dim': 32,
  'num_layers': 3,
  'course': 'BBB',
  'num_weeks': 15,
  'num_classes': 2},
 {'epochs': 100,
  'hidden_dim': 128,
  'learning_rate': 0.1,
  'dropout': 0.5,
  'weight_decay': 0.0001,
  'fcn_dim': 16,
  'num_layers': 2,
  'course': 'BBB',
  'num_weeks': 20,
  'num_classes': 2},
 {'epochs': 100,
  'hidden_dim': 128,
  'learning_rate': 0.001,
  'dropout': 0.3,
  'weight_decay': 0.0001,
  'fcn_dim': 16,
  'num_layers': 3,
  'course': 'DDD',
  'num_weeks': 5,
  'num_classes': 2},
 {'epochs': 100,
  'hidden_dim': 64,
  'learning_rate': 0.001,
  'dropout': 0.3,
  'weight_decay': 0.0001,
  'fcn_dim': 64,
  'num_layers': 2,
  'course': 'DDD',
  'num_weeks': 10,
  'num_classes': 2},
 {'epochs': 200,
  'hidden_dim': 128,
  'learning_rate': 1.0,
  'dropout': 0.3,
  'weight_decay': 0.0001,
  'fcn_dim': 64,
  'num_layers': 3,
  'course': 'DDD',
  'num_weeks': 15,
  'num_classes': 2},
 {'epochs': 200,
  'hidden_dim': 128,
  'learning_rate': 0.001,
  'dropout': 0.5,
  'weight_decay': 0.0001,
  'fcn_dim': 32,
  'num_layers': 3,
  'course': 'DDD',
  'num_weeks': 20,
  'num_classes': 2},
 {'epochs': 100,
  'hidden_dim': 64,
  'learning_rate': 0.001,
  'dropout': 0.2,
  'weight_decay': 0.0001,
  'fcn_dim': 64,
  'num_layers': 3,
  'course': 'EEE',
  'num_weeks': 5,
  'num_classes': 2},
 {'epochs': 200,
  'hidden_dim': 64,
  'learning_rate': 0.001,
  'dropout': 0.2,
  'weight_decay': 0.0001,
  'fcn_dim': 16,
  'num_layers': 3,
  'course': 'EEE',
  'num_weeks': 10,
  'num_classes': 2},
 {'epochs': 200,
  'hidden_dim': 64,
  'learning_rate': 0.001,
  'dropout': 0.2,
  'weight_decay': 0.0001,
  'fcn_dim': 16,
  'num_layers': 2,
  'course': 'EEE',
  'num_weeks': 15,
  'num_classes': 2},
 {'epochs': 100,
  'hidden_dim': 64,
  'learning_rate': 0.001,
  'dropout': 0.5,
  'weight_decay': 0.0001,
  'fcn_dim': 16,
  'num_layers': 2,
  'course': 'EEE',
  'num_weeks': 20,
  'num_classes': 2},
 {'epochs': 200,
  'hidden_dim': 64,
  'learning_rate': 0.001,
  'dropout': 0.5,
  'weight_decay': 0.0001,
  'fcn_dim': 16,
  'num_layers': 2,
  'course': 'FFF',
  'num_weeks': 5,
  'num_classes': 2},
 {'epochs': 100,
  'hidden_dim': 128,
  'learning_rate': 0.001,
  'dropout': 0.3,
  'weight_decay': 0.0001,
  'fcn_dim': 16,
  'num_layers': 2,
  'course': 'FFF',
  'num_weeks': 10,
  'num_classes': 2},
 {'epochs': 200,
  'hidden_dim': 128,
  'learning_rate': 1.0,
  'dropout': 0.2,
  'weight_decay': 0.0001,
  'fcn_dim': 64,
  'num_layers': 2,
  'course': 'FFF',
  'num_weeks': 15,
  'num_classes': 2},
 {'epochs': 200,
  'hidden_dim': 64,
  'learning_rate': 0.5,
  'dropout': 0.3,
  'weight_decay': 0.0001,
  'fcn_dim': 16,
  'num_layers': 2,
  'course': 'FFF',
  'num_weeks': 20,
  'num_classes': 2},
 {'epochs': 100,
  'hidden_dim': 128,
  'learning_rate': 0.01,
  'dropout': 0.5,
  'weight_decay': 0.0001,
  'fcn_dim': 64,
  'num_layers': 3,
  'course': 'GGG',
  'num_weeks': 5,
  'num_classes': 2},
 {'epochs': 100,
  'hidden_dim': 64,
  'learning_rate': 0.01,
  'dropout': 0.5,
  'weight_decay': 0.0001,
  'fcn_dim': 16,
  'num_layers': 3,
  'course': 'GGG',
  'num_weeks': 10,
  'num_classes': 2},
 {'epochs': 100,
  'hidden_dim': 64,
  'learning_rate': 0.01,
  'dropout': 0.3,
  'weight_decay': 0.0001,
  'fcn_dim': 16,
  'num_layers': 2,
  'course': 'GGG',
  'num_weeks': 15,
  'num_classes': 2},
 {'epochs': 200,
  'hidden_dim': 64,
  'learning_rate': 0.001,
  'dropout': 0.3,
  'weight_decay': 0.0001,
  'fcn_dim': 16,
  'num_layers': 3,
  'course': 'GGG',
  'num_weeks': 20,
  'num_classes': 2}]

class LSTM_atten_emb(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, fcn_dim, num_classes, dropout=0.5):
        super(LSTM_atten_emb, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout = dropout
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.attention = Attention(hidden_dim)
        self.fcn = nn.Sequential(nn.Linear(hidden_dim, fcn_dim), nn.ReLU())
        if num_classes == 2:
            self.linear = nn.Linear(fcn_dim, 1)
        else:
            self.linear = nn.Linear(fcn_dim, num_classes)

    def forward(self, input_data):
        lstm_out, _ = self.lstm(input_data)
        attention_out = self.attention(lstm_out)  # attention_out.shape:  torch.Size([100, 64])
        # attention_out = torch.cat([attention_out, embed], dim=1)
        attention_out = self.fcn(attention_out)
        logits = self.linear(attention_out)
        if num_classes == 2:
            logits = torch.sigmoid(logits)
        return logits


class Attention(nn.Module):
    def __init__(self, input_dim):
        super(Attention, self).__init__()
        self.input_dim = input_dim
        self.linear_out = nn.Linear(input_dim, 1)

    def forward(self, input_data):
        bs, sl, dim = input_data.shape
        x = input_data.reshape(-1, dim)
        x = self.linear_out(x)
        x = x.view(bs, sl)
        x = F.softmax(x, dim=1)
        weighted = input_data * x.unsqueeze(-1)
        return weighted.sum(dim=1)


df = pd.read_csv('./data/clickstream_dataset_updated.csv')
df.columns = range(df.columns.size)
df.rename({0: 'timestamp', 1: 'course_id', 2: 'user_id'}, axis=1, inplace=True)
courses = pd.unique(df['course_id'])
# demo_length = 37
# weeks = [5]#, 10, 15, 20]

input_size = 20
num_classes = args.num_classes
device = torch.device("cuda")
# courses = ['BBB']
course = args.course
num_weeks = args.num_weeks
# for course in courses:
file1 = open(f"./final_results/{course}_{num_weeks}_weeks_lstm_grid_search_results.txt", "a")
file2 = open(f"./final_results/{course}_{num_weeks}_weeks_lstm_validation_f1.txt", "a")

df_train = pd.read_csv(f'./saved_data/baseline/train_phase/{course}_{num_classes}/train_dataset.csv')
df_train.columns = range(df_train.columns.size)
df_test = pd.read_csv(f'./saved_data/baseline/train_phase/{course}_{num_classes}/test_dataset.csv')
df_test.columns = range(df_test.columns.size)
df_train = df_train.drop(range(600, 637), axis=1)

df_test = df_test.drop(range(600, 637), axis=1)

df_dev = pd.read_csv(f'./saved_data/baseline/train_phase/{course}_{num_classes}/dev_dataset.csv')
df_dev.columns = range(df_dev.columns.size)
df_dev = df_dev.drop(range(600, 637), axis=1)

    # df_train = pd.concat([df_train, df_dev], ignore_index=True)

# for num_weeks in weeks:
print(f'Predicting for course {course} with {num_weeks} weeks.')
# file1.write(f"Predicting for course {course} with {num_weeks} weeks.")

# json.dump(args.__dict__, file1, indent=2)
file1.write(json.dumps(args.__dict__))
file2.write(json.dumps(args.__dict__))
h_size = args.hidden_dim
num_layers = args.num_layers
dropout = args.dropout
weight_decay = args.weight_decay
epochs = args.epochs
fcn_dim = args.fcn_dim
lr = args.learning_rate

model = LSTM_atten_emb(input_dim=input_size, hidden_dim=h_size, fcn_dim=fcn_dim, num_layers=num_layers,
                       num_classes=num_classes, dropout=dropout).to(device)
prev_epoch = 0
try:
    # Find all the name of the model files
    model_files = glob.glob(f"/models/lstm_models/{course}_{num_weeks}_weeks_{str(model.__class__.__name__)}_model*.pth")
    # Extract the epoc number from the model name
    prev_epoch = int(re.findall(r'\d+', model_files[0])[0])
    prev_lr = float("0." + re.findall(r'\d+', model_files[0])[2])
    prev_acc = float(re.findall(r'\d+', model_files[0])[5]) / 100
    print(
        f"Loading {str(model.__class__.__name__)} for {num_weeks} weeks of course {course} trained for {prev_epoch} with LR {prev_lr} achieving {prev_acc} named {model_files}")
    model.load_state_dict(torch.load(model_files[0]))
except:
    pass

# Set loss and optimizer function
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
# Best loss function for multi-class classification
criterion = nn.BCELoss()

scaler = StandardScaler()
train_x = df_train.iloc[:, :num_weeks * input_size].values
train_y = df_train.iloc[:, -1:].values

train_x = scaler.fit_transform(train_x)
print(train_x.shape)
print(train_y.shape)
train_x = np.array(train_x, dtype=np.float32)

train_x = train_x.reshape(-1, num_weeks, input_size)
train_x = torch.from_numpy(train_x).float().to(device)
train_y = torch.from_numpy(train_y).long().to(device)

train_loader = DataLoader(
    dataset=list(zip(train_x, train_y)), batch_size=100, shuffle=True
)

timestamp = time.time()
file1.write(f"\nModel timestamp: {timestamp}\n")
file2.write(f"\nModel timestamp: {timestamp}\n")
# f = open(f'./lstm_logs/{course}_{num_weeks}_weeks_lstm_train_log_{timestamp}.txt', 'w')
prev_acc = 0
for epoch in range(epochs):
    total = 0
    correct = 0
    for X, y in tqdm(train_loader, total=len(train_loader)):
        optimizer.zero_grad()
        # outputs = model(train_x)
        outputs = model(X)
        loss = criterion(outputs.to(torch.float32), y.to(torch.float32))
        loss.backward()
        optimizer.step()

        # find accuracy
        if num_classes == 2:
            predicted = (outputs.data > 0.5).int()
        else:
            _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == y).sum()
        total += y.size(0)

    accuracy = correct / total
    if (epoch + 1) % 1 == 0:
        result = "Epoch: [%d/%d], LR: %.7f, Accuracy: %.4f \n" % (
            epoch + 1, epochs, optimizer.param_groups[0]['lr'], accuracy)
        print(result)

        # f.write(result)

    if prev_acc < accuracy:
        # Export the model
        # Remove the previous model
        if glob.glob("models/lstm_models/*.pth") != []:
            try:
                os.remove(
                    glob.glob(f"models/lstm_models/{course}_{num_weeks}_weeks_{str(model.__class__.__name__)}_model*.pth")[
                        0])
            except:
                pass
        torch.save(model.state_dict(),
                   f"models/lstm_models/{course}_{num_weeks}_weeks_{str(model.__class__.__name__)}_model_{epoch + prev_epoch}_{optimizer.param_groups[0]['lr']}_{accuracy:.4f}_{timestamp}.pth")
        prev_acc = accuracy
    else:
        scheduler.step()

# Read the text in ast_GNN.txt as a string
# with open(f'./lstm_logs/{course}_{num_weeks}_weeks_lstm_train_log_{timestamp}.txt', "r") as f:
#     string = f.read()
#
# acc_values = re.findall(r"Accuracy: (\d+\.\d+)", string)
# acc_values = [float(x) for x in acc_values]
#
# # Plot the accuracy values over the epochs
# plt.figure(figsize=(10, 5))
# plt.plot(acc_values)
# # Plot the moving average of the accuracy values
# plt.plot(np.convolve(acc_values, np.ones((20,)) / 20, mode='valid'))
# plt.xlabel("Epochs")
# plt.ylabel("Accuracy")
# plt.show()


def evaluate_model(dev_x, dev_y, type='Validation'):
    dev_x = scaler.transform(dev_x)
    print(dev_x.shape)
    print(dev_y.shape)

    dev_x = np.array(dev_x, dtype=np.float32)

    dev_x = dev_x.reshape(-1, num_weeks, input_size)

    dev_x = torch.from_numpy(dev_x).float().cuda()
    dev_y = torch.from_numpy(dev_y).long().cuda()

    test_loader = DataLoader(
        dataset=list(zip(dev_x, dev_y)), batch_size=len(dev_y), shuffle=True
    )

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for dev_x, dev_y in tqdm(test_loader, total=len(test_loader)):
            # outputs = model(dev_x)
            outputs = model(dev_x)
            if num_classes == 2:
                predicted = (outputs.data > 0.5).int()
            else:
                _, predicted = torch.max(outputs.data, 1)
            # _, predicted = torch.max(outputs.data, 1)
            total += dev_y.size(0)
            correct += (predicted == dev_y).sum()
            # Classificaiton report
        print(f'\n{type} results:')
        file1.write(f'\n{type} results:')
        print(classification_report(dev_y.cpu(), predicted.cpu()))
        if type == 'Validation':
            f1_value = f1_score(dev_y.cpu(), predicted.cpu(), average='weighted')
            file2.write(f'F1-score: {str(f1_value)}\n')
        file1.write('\n' + str(classification_report(dev_y.cpu(), predicted.cpu())))
        print(f"{type} Accuracy of the model on the test data: {(100 * correct / total)}")


# if course not in ['AAA', 'CCC']:
dev_x = df_dev.iloc[:, :num_weeks * input_size].values
dev_y = df_dev.iloc[:, -1:].values

evaluate_model(dev_x, dev_y)

test_x = df_test.iloc[:, :num_weeks * input_size].values
test_y = df_test.iloc[:, -1:].values
evaluate_model(test_x, test_y, type='Test')
# else:
#     test_x = df_test.iloc[:, :num_weeks * input_size].values
#     test_y = df_test.iloc[:, -1:].values
#     evaluate_model(test_x, test_y, type='Test')

file1.close()
file2.close()