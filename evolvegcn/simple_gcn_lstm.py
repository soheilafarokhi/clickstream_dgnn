from torch import nn
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math

class GCN_LSTM(nn.Module):
    def __init__(self, args, activation, dropout=0.2, device='cpu', use_gru=False):
        super(GCN_LSTM, self).__init__()
        feats = [args.feats_per_node,
                 args.layer_1_feats,
                 args.layer_2_feats]
        self.device = device
        self.GCN_layers = []
        self._parameters = nn.ParameterList()
        self.dropout = dropout
        # for i in range(1, len(feats)):
        #     GCN_args = u.Namespace({'in_feats': feats[i - 1],
        #                              'out_feats': feats[i],
        #                              'activation': activation})
        #
        #     gcn_i = GCN(GCN_args)
        #     # print (i,'grcu_i', grcu_i)
        #     self.GCN_layers.append(gcn_i.to(self.device))
        #     self._parameters.extend(list(self.GCN_layers[-1].parameters()))
        self.gc1 = GCN(feats[0], feats[1]).to(self.device)
        self._parameters.extend(list(self.gc1.parameters()))
        self.gc2 = GCN(feats[1], feats[2]).to(self.device)
        self._parameters.extend(list(self.gc2.parameters()))
        self.lstm = LSTM_atten_emb(use_gru=use_gru, input_dim=(feats[2] + 20), hidden_dim=args.lstm_l2_hidden_dim, num_layers=args.lstm_l2_layers,
                                   lstm_output_dim=args.lstm_l2_feats,  dropout=args.lstm_dropout).to(self.device)
        self._parameters.extend(list(self.lstm.parameters()))

    def parameters(self):
        return self._parameters

    def forward(self, adj_list, nodes_list, mask_list, raw_data_list=None):

        for t, adj_hat in enumerate(adj_list):

            node_embs = nodes_list[t]
            x = F.relu(self.gc1(node_embs, adj_hat))
            x = F.dropout(x, self.dropout, training=self.training)
            embed = self.gc2(x, adj_hat)
            # embed = F.log_softmax(x, dim=1)
            if raw_data_list is not None:
                raw_data = raw_data_list[t]
                embed = torch.cat((embed, raw_data), 1)
                embed = embed.view(-1, 1, self.gc2.out_features + raw_data.shape[1])
            else:
                embed = embed.view(-1, 1, self.gc2.out_features)
            if t == 0:
                out_seq = embed.to(self.device)
            else:
                out_seq = torch.cat((out_seq,embed), dim=1)

        output = self.lstm(out_seq)
        # out = nodes_list
        return output



class GCN(nn.Module):
    def __init__(self, in_feats, out_feats, bias=True):
        super(GCN, self).__init__()

        self.in_features = in_feats
        self.out_features = out_feats
        self.weight = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(self.out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()


    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class LSTM_atten_emb(nn.Module):
    def __init__(self, use_gru, input_dim, hidden_dim, num_layers, lstm_output_dim, dropout=0.5):
        super(LSTM_atten_emb, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm_output_dim = lstm_output_dim
        self.dropout = dropout
        if use_gru:
            self.lstm = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        else:
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.attention = Attention(hidden_dim)
        self.fcn = nn.Sequential(nn.Linear(hidden_dim, 128), nn.ReLU())
        self.linear = nn.Linear(128, lstm_output_dim)

    def forward(self, input_data):
        lstm_out, _ = self.lstm(input_data)
        attention_out = self.attention(lstm_out)  # attention_out.shape:  torch.Size([100, 64])
        # attention_out = torch.cat([attention_out, embed], dim=1)
        attention_out = self.fcn(attention_out)
        logits = self.linear(attention_out)
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
