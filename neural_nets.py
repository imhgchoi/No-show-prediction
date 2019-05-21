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

        def init_weights(m):
            if type(m) == nn.Linear:
                T.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

        self.layer1 = nn.Sequential(nn.Linear(26, 48, bias=True), nn.Dropout(0.2))
        self.layer1.apply(init_weights)
        self.bn1 = nn.BatchNorm1d(48)
        self.layer2 = nn.Sequential(nn.Linear(48, 24, bias=True), nn.Dropout(0.2))
        self.layer2.apply(init_weights)
        self.bn2 = nn.BatchNorm1d(24)
        self.layer3 = nn.Sequential(nn.Linear(24, 10, bias=True), nn.Dropout(0.2))
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

        self.optimizer = optim.Adam(self.parameters(), lr=self.config.lr)
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