import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

class MLP(nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()
        self.config = config
        if config.impt_only :
            self.input_dim = 20
        else :
            self.input_dim = 26

        def init_weights(m):
            if type(m) == nn.Linear:
                T.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

        self.hidden = nn.Sequential(nn.Linear(self.input_dim, 50, bias=True))
        self.hidden.apply(init_weights)
        if self.config.output_activation == 'sigmoid' :
            self.out = nn.Linear(50, 1)
            self.out.apply(init_weights)
            self.loss = nn.MSELoss()
        else :
            self.out = nn.Linear(50, 2)
            self.out.apply(init_weights)
            self.loss = nn.CrossEntropyLoss()

        self.optimizer = optim.SGD(self.parameters(), lr=self.config.lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, data):
        hidden = self.hidden(data)
        out = self.out(hidden)
        if self.config.output_activation == 'sigmoid' :
            return T.sigmoid(out)
        else :
            return T.softmax(out, dim=1)


class DNN(nn.Module):
    def __init__(self, config):
        super(DNN, self).__init__()
        self.config = config
        if config.impt_only :
            self.input_dim = 20
        else :
            self.input_dim = 26

        def init_weights(m):
            if type(m) == nn.Linear:
                T.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

        self.layer1 = nn.Sequential(nn.Linear(self.input_dim, 100, bias=True), nn.Dropout(0.1))
        self.layer1.apply(init_weights)
        self.bn1 = nn.BatchNorm1d(100)
        self.layer2 = nn.Sequential(nn.Linear(100, 50, bias=True), nn.Dropout(0.1))
        self.layer2.apply(init_weights)
        self.bn2 = nn.BatchNorm1d(50)
        self.layer3 = nn.Sequential(nn.Linear(50, 10, bias=True), nn.Dropout(0.1))
        self.layer3.apply(init_weights)
        self.bn3 = nn.BatchNorm1d(10)
        if self.config.output_activation == 'sigmoid' :
            self.out = nn.Linear(10, 1)
            self.out.apply(init_weights)
            self.loss = nn.MSELoss()
        else :
            self.out = nn.Linear(10, 2)
            self.out.apply(init_weights)
            self.loss = nn.CrossEntropyLoss()

        self.optimizer = optim.SGD(self.parameters(), lr=self.config.lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, data):
        layer1 = self.bn1(F.relu(self.layer1(data)))
        layer2 = self.bn2(F.relu(self.layer2(layer1)))
        layer3 = self.bn3(F.relu(self.layer3(layer2)))
        out = self.out(layer3)
        if self.config.output_activation == 'sigmoid' :
            return T.sigmoid(out)
        else :
            return T.softmax(out, dim=1)