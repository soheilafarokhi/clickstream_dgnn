import utils as u
import torch
from torch.nn.parameter import Parameter
import torch.nn as nn
import math
import torch.nn.functional as F

class EGCN_LSTM(torch.nn.Module):
    def __init__(self, args, activation, device='cpu', skipfeats=False, use_gru=False):
        super().__init__()
        # GRCU_args = u.Namespace({})

        feats = [args.feats_per_node,
                 args.layer_1_feats,
                 args.layer_2_feats]
        self.device = device
        self.skipfeats = skipfeats
        self.GRCU_layers = []
        self._parameters = nn.ParameterList()
        self.gcn_out_feats = args.layer_2_feats

        for i in range(1, len(feats)):
            GRCU_args = u.Namespace({'in_feats': feats[i - 1],
                                     'out_feats': feats[i],
                                     'activation': activation})

            grcu_i = GRCU(GRCU_args)
            # print (i,'grcu_i', grcu_i)
            self.GRCU_layers.append(grcu_i.to(self.device))
            self._parameters.extend(list(self.GRCU_layers[-1].parameters()))

        self.lstm = LSTM_atten_emb(use_gru=use_gru, input_dim=(feats[2] + 20), hidden_dim=args.lstm_l2_hidden_dim, num_layers=args.lstm_l2_layers,
                                   lstm_output_dim=args.lstm_l2_feats, dropout=args.lstm_dropout).to(self.device)
        self._parameters.extend(list(self.lstm.parameters()))
    def parameters(self):
        return self._parameters

    def forward(self, adj_list, nodes_list, nodes_mask_list, raw_data_list=None):
        node_feats = nodes_list[-1]

        for unit in self.GRCU_layers:
            nodes_list = unit(adj_list, nodes_list, nodes_mask_list)


        for t, adj_hat in enumerate(adj_list):

            embed = nodes_list[t]

            if raw_data_list is not None:
                raw_data = raw_data_list[t]
                embed = torch.cat((embed, raw_data), 1)
                embed = embed.view(-1, 1, self.gcn_out_feats + raw_data.shape[1])
            else:
                embed = embed.view(-1, 1, self.gcn_out_feats)
            if t == 0:
                out_seq = embed.to(self.device)
            else:
                out_seq = torch.cat((out_seq,embed), dim=1)

        out = self.lstm(out_seq)

        # if self.skipfeats:
        #     out = torch.cat((out[-1], node_feats), dim=1)  # use node_feats.to_dense() if 2hot encoded input
        return out


class GRCU(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        cell_args = u.Namespace({})
        cell_args.rows = args.in_feats
        cell_args.cols = args.out_feats

        self.evolve_weights = mat_GRU_cell(cell_args)

        self.activation = self.args.activation
        self.GCN_init_weights = Parameter(torch.Tensor(self.args.in_feats, self.args.out_feats))
        self.reset_param(self.GCN_init_weights)

    def reset_param(self, t):
        # Initialize based on the number of columns
        stdv = 1. / math.sqrt(t.size(1))
        t.data.uniform_(-stdv, stdv)

    def forward(self, adj_list, node_embs_list, mask_list):
        GCN_weights = self.GCN_init_weights
        out_seq = []
        for t, adj_hat in enumerate(adj_list):
            node_embs = node_embs_list[t]
            # first evolve the weights from the initial and use the new weights with the node_embs
            GCN_weights = self.evolve_weights(GCN_weights, node_embs, mask_list[t])
            node_embs = self.activation(adj_hat.matmul(node_embs.matmul(GCN_weights)))

            out_seq.append(node_embs)

        return out_seq


class mat_GRU_cell(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.update = mat_GRU_gate(args.rows,
                                   args.cols,
                                   torch.nn.Sigmoid())

        self.reset = mat_GRU_gate(args.rows,
                                  args.cols,
                                  torch.nn.Sigmoid())

        self.htilda = mat_GRU_gate(args.rows,
                                   args.cols,
                                   torch.nn.Tanh())

        self.choose_topk = TopK(feats=args.rows,
                                k=args.cols)

    def forward(self, prev_Q, prev_Z, mask):
        z_topk = self.choose_topk(prev_Z, mask)

        update = self.update(z_topk, prev_Q)
        reset = self.reset(z_topk, prev_Q)

        h_cap = reset * prev_Q
        h_cap = self.htilda(z_topk, h_cap)

        new_Q = (1 - update) * prev_Q + update * h_cap

        return new_Q


class mat_GRU_gate(torch.nn.Module):
    def __init__(self, rows, cols, activation):
        super().__init__()
        self.activation = activation
        # the k here should be in_feats which is actually the rows
        self.W = Parameter(torch.Tensor(rows, rows))
        self.reset_param(self.W)

        self.U = Parameter(torch.Tensor(rows, rows))
        self.reset_param(self.U)

        self.bias = Parameter(torch.zeros(rows, cols))

    def reset_param(self, t):
        # Initialize based on the number of columns
        stdv = 1. / math.sqrt(t.size(1))
        t.data.uniform_(-stdv, stdv)

    def forward(self, x, hidden):
        out = self.activation(self.W.matmul(x) + \
                              self.U.matmul(hidden) + \
                              self.bias)

        return out


class TopK(torch.nn.Module):
    def __init__(self, feats, k):
        super().__init__()
        self.scorer = Parameter(torch.Tensor(feats, 1))
        self.reset_param(self.scorer)

        self.k = k

    def reset_param(self, t):
        # Initialize based on the number of rows
        stdv = 1. / math.sqrt(t.size(0))
        t.data.uniform_(-stdv, stdv)

    def forward(self, node_embs, mask):
        scores = node_embs.matmul(self.scorer) / self.scorer.norm()
        scores = scores + mask

        vals, topk_indices = scores.view(-1).topk(self.k)
        topk_indices = topk_indices[vals > -float("Inf")]

        if topk_indices.size(0) < self.k:
            topk_indices = u.pad_with_last_val(topk_indices, self.k)

        tanh = torch.nn.Tanh()

        if isinstance(node_embs, torch.sparse.FloatTensor) or \
                isinstance(node_embs, torch.cuda.sparse.FloatTensor):
            node_embs = node_embs.to_dense()

        out = node_embs[topk_indices] * tanh(scores[topk_indices].view(-1, 1))

        # we need to transpose the output
        return out.t()


class LSTM_atten_emb(nn.Module):
    def __init__(self, use_gru, input_dim, hidden_dim, num_layers, lstm_output_dim, dropout=0.1):
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
