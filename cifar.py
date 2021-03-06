import torch
import torch.nn as nn
import numpy as np
import math
import torchvision
from sklearn.utils import shuffle
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from matplotlib import pyplot as plt
import copy
from torchvision.transforms import transforms

class ResBasicBlock(nn.Module):
    EXPANSION = 1

    def __init__(self, in_planes, planes, stride=1):
        super(ResBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.EXPANSION * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.EXPANSION * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.EXPANSION * planes)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Bottleneck(nn.Module):
    EXPANSION = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, self.EXPANSION * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.EXPANSION * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.EXPANSION * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.EXPANSION * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.EXPANSION * planes)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        woha = self.shortcut(x)
        out += woha
        out = F.relu(out)
        return out

class ResNet(nn.Module):

    def __init__(self, block, num_blocks, num_classes=10):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        super(ResNet, self).__init__()
        self.num_blocks = num_blocks
        self.in_planes = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.num_classes = num_classes
        self.hidden_layers = []
        self.output_layers = []

        self._make_layer(block, 64, num_blocks[0], stride=1)

        self._make_layer(block, 128, num_blocks[1], stride=2)

        self._make_layer(block, 256, num_blocks[2], stride=2)

        self._make_layer(block, 512, num_blocks[3], stride=2)


        self.output_layers.append(self._make_mlp1(64, 2))  # 32
        self.output_layers.append(self._make_mlp1(64, 2))  # 32
        self.output_layers.append(self._make_mlp2(128, 2))  # 16
        self.output_layers.append(self._make_mlp2(128, 2))  # 16
        self.output_layers.append(self._make_mlp3(256, 2))  # 8
        self.output_layers.append(self._make_mlp3(256, 2))  # 8
        self.output_layers.append(self._make_mlp4(512, 2))  # 4
        self.output_layers.append(self._make_mlp4(512, 2))  # 4

        self.hidden_layers = nn.ModuleList(self.hidden_layers)  #
        self.output_layers = nn.ModuleList(self.output_layers)  #

    def _make_mlp1(self, in_planes,  kernel_size_pool, padding_pool=0):
        classifier = nn.Sequential(
            nn.MaxPool2d(kernel_size=kernel_size_pool, padding=padding_pool),
            nn.MaxPool2d(kernel_size=kernel_size_pool, padding=padding_pool),
            #nn.MaxPool2d(kernel_size=kernel_size_pool, padding=padding_pool),
            nn.Flatten(),
            nn.Linear(in_planes*8*8, in_planes*8*2),
            nn.Linear(in_planes*8*2, 256),
            nn.Linear(256, self.num_classes),
        )
        return classifier
    def _make_mlp2(self, in_planes, kernel_size_pool, padding_pool=0):
        classifier = nn.Sequential(
            nn.MaxPool2d(kernel_size=kernel_size_pool, padding=padding_pool,ceil_mode=True),
            nn.MaxPool2d(kernel_size=kernel_size_pool, padding=padding_pool,ceil_mode=True),
            # nn.MaxPool2d(kernel_size=kernel_size_pool, padding=padding_pool),
            nn.Flatten(),
            nn.Linear(in_planes*4*4, self.num_classes),
        )
        return classifier
    def _make_mlp3(self, in_planes, kernel_size_pool, padding_pool=0):
        classifier = nn.Sequential(
            nn.MaxPool2d(kernel_size=kernel_size_pool, padding=padding_pool,ceil_mode=True),
            nn.AvgPool2d(kernel_size=kernel_size_pool, padding=padding_pool,ceil_mode=True),
            # nn.MaxPool2d(kernel_size=kernel_size_pool, padding=padding_pool),
            nn.Flatten(),
            nn.Linear(in_planes*2*2, self.num_classes),
        )
        return classifier

    def _make_mlp4(self, in_planes, kernel_size_pool, padding_pool=0):
        classifier = nn.Sequential(
            nn.AvgPool2d(kernel_size=kernel_size_pool, padding=padding_pool,ceil_mode=True),
            # nn.MaxPool2d(kernel_size=kernel_size_pool, padding=padding_pool),
            nn.Flatten(),
            nn.Linear(in_planes*2*2, self.num_classes),
        )
        return classifier



    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        for stride in strides:
            self.hidden_layers.append(block(self.in_planes, planes, stride))
            self.in_planes = block.EXPANSION * planes

    def forward(self, x):
        hidden_connections = []
        hidden_connections.append(F.relu(self.bn1(self.conv1(x))))

        for i in range(len(self.hidden_layers)):
            hidden_connections.append(self.hidden_layers[i](hidden_connections[i]))

        output_class = []
        for i in range(len(self.output_layers)):
            #print(hidden_connections[i].shape)
            output_class.append(self.output_layers[i](hidden_connections[i]))

        return output_class

class BasicBlock(nn.Module):    # BasicBlock from ResNet [He et al.2016]
    EXPANSION = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.EXPANSION*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.EXPANSION*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.EXPANSION*planes)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder,self).__init__()
        self.conv1 = nn.Conv2d(3, 12, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(12)
        self.conv2 = nn.ConvTranspose2d(12, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(1)
        self.encoder = BasicBlock(12, 1)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1, 12, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.ConvTranspose2d(12, 12, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.ConvTranspose2d(12, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(3)
        )

    def forward(self, x):

        encoded = self.encoder(F.relu(self.bn1(self.conv1(x))))     # maps the feature size from 3*32*32 to 32*32
        decoded = self.decoder(encoded)
        return encoded, decoded

def Dynamic_ResNet18():
    return ResNet(ResBasicBlock, [1, 2, 2, 2])

class cifar:
    def __init__(self, data_S1, label_S1, data_S2, label_S2, T1, t, b=0.9, lr=0.01, s=0.008, m=0.95, spike=9e-5,
                 thre=10000):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.autoencoder = AutoEncoder().to(self.device)
        self.autoencoder_2 = AutoEncoder().to(self.device)
        self.spike = spike  # setting a spike factor, expedite convergence to deep layers with few shots
        self.thre = thre  # threshold for spiking
        self.beta1 = 1  # subtotal weight for shallow layers
        self.beta2 = 0  # subtotal weight for deep layers
        self.correct = 0
        self.accuracy = 0
        self.T1 = T1  # for data come from S1
        self.t = t
        self.B = self.T1 - self.t  # start of an evolving period of time
        self.data_S1 = data_S1
        self.label_S1 = label_S1
        self.data_S2 = data_S2
        self.label_S2 = label_S2
        self.b = Parameter(torch.tensor(b), requires_grad=False).to(self.device)
        self.lr = Parameter(torch.tensor(lr), requires_grad=False).to(self.device)
        self.s = Parameter(torch.tensor(s), requires_grad=False).to(self.device)
        self.m = Parameter(torch.tensor(m), requires_grad=False).to(self.device)
        self.s1 = Parameter(torch.tensor(0.01), requires_grad=False).to(self.device)
        self.num_block = 8  # num of dynamic building blocks
        self.alpha1 = Parameter(torch.Tensor(self.num_block).fill_(1 / self.num_block), requires_grad=False).to(
            self.device)
        self.alpha2 = Parameter(torch.Tensor(self.num_block).fill_(1 / self.num_block), requires_grad=False).to(
            self.device)
        self.CELoss = nn.CrossEntropyLoss()
        self.BCELoss = nn.BCELoss()
        "For plot"
        self.S1_lossFigure_i = []
        self.S1_Figure_Accuracy = []
        self.alpha_array = []

        self.a_1 = 0.8
        self.a_2 = 0.2
        self.cl_1 = []
        self.cl_2 = []

    def T_1(self):  # Feature Evolving Streaming Approximate
        data1 = self.data_S1  # B donate the cashe begins time and B+t donate during the cashe time
        data2 = self.data_S2[self.B:]
        self.net_model1 = Dynamic_ResNet18().to(self.device)
        self.CELoss = nn.CrossEntropyLoss()
        self.BCEloss = nn.BCELoss()
        self.SmoothL1Loss = nn.SmoothL1Loss()
        optimizer_1 = torch.optim.SGD(self.net_model1.parameters(), lr=0.01)
        optimizer_2 = torch.optim.SGD(self.autoencoder.parameters(), lr=0.02)
        self.net_model1.to(self.device)
        b = -0.01
        for (i, x) in enumerate(data1):
            self.i = i
            x1 = x.unsqueeze(0).float().to(self.device)
            self.i = i
            y = self.label_S1[i].unsqueeze(0).long().to(self.device)
            if self.i < self.B:  # Before evolve
                encoded, decoded = self.autoencoder(x1)
                optimizer_2.zero_grad()
                loss_1, y_hat = self.HB_Fit(self.net_model1, encoded, y, optimizer_1)

                loss_2 = self.BCEloss(torch.sigmoid(decoded), x1)
             

                loss_2.backward()

                optimizer_2.step()

                if self.i < self.thre:  # Spike!
                    self.beta2 = self.beta2 + self.spike
                    self.beta1 = 1 - self.beta2


            else:  # Evolving start
                x2 = data2[self.i - self.B].unsqueeze(0).float().to(self.device)
                if i == self.B:
                    self.net_model2 = copy.deepcopy(self.net_model1)
                    torch.save(self.net_model1.state_dict(),
                               './data/net_model1.pth')
                    optimizer_1_1 = torch.optim.SGD(self.net_model1.parameters(), lr=0.01)
                    optimizer_1_2 = torch.optim.SGD(self.net_model2.parameters(), lr=0.01)
                    optimizer_2_1 = torch.optim.SGD(self.autoencoder.parameters(), lr=0.02)
                    optimizer_2_2 = torch.optim.SGD(self.autoencoder_2.parameters(), lr=0.08)

                encoded_1, decoded_1 = self.autoencoder(x1)  # torch.relu(self.conv1(x1))
                encoded_2, decoded_2 = self.autoencoder_2(x2)

                optimizer_2_2.zero_grad()
                loss_1_1, y_hat_1 = self.HB_Fit(self.net_model1, encoded_1, y, optimizer_1_1)
                loss_1_2, y_hat_2 = self.HB_Fit(self.net_model2, encoded_2, y, optimizer_1_2)

                y_hat = self.a_1 * y_hat_1 + self.a_2 * y_hat_2

                self.cl_1.append(loss_1_1)
                self.cl_2.append(loss_1_2)

                if len(self.cl_1) == 100:
                    self.cl_1.pop(0)
                    self.cl_2.pop(0)

                try:
                    a_cl_1 = math.exp(b * sum(self.cl_1))
                    a_cl_2 = math.exp(b * sum(self.cl_2))
                    self.a_1 = (a_cl_1) / (a_cl_2 + a_cl_1)

                except OverflowError:
                    self.a_1 = float('inf')

                self.a_2 = 1 - self.a_1

                loss_2_2 = self.BCEloss(torch.sigmoid(decoded_2), x2)
                loss_2_0 = self.SmoothL1Loss(encoded_2,
                                             encoded_1)  # Regularizer to enforce the latent representations' similarity
                loss_2 = loss_2_0 + loss_2_2
                loss_2.backward(retain_graph=True)
                optimizer_2_2.step()

            _, predicted = torch.max(y_hat.data, 1)
            self.correct += (predicted == y).item()

            if i == 0:
                print("finish 0")
            if (i + 1) % 100 == 0:
                print("step : %d" % (i + 1), end=", ")
                print("correct: %d" % (self.correct))
            if (i + 1) % 1000 == 0:
                self.accuracy = self.correct / 1000
                self.S1_Figure_Accuracy.append(self.accuracy)
                self.alpha_array.append(self.alpha1)
                self.correct = 0
                print("Accuracy: ", self.accuracy)
                print(self.alpha1)

        torch.save(self.net_model2.state_dict(),
                   './data/net_model2.pth')

    def T_2(self):
        print("start from T1 when i<T1 ")
        self.T_1()
        self.correct = 0
        "load data for T2"
        data2 = self.data_S2[:self.B]
        "use the Network"
        net_model1 = Dynamic_ResNet18().to(self.device)
        pretrain_dict = torch.load(
            './data/net_model1.pth')  # One model, no ensembling
        model_dict = net_model1.state_dict()
        pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict}
        model_dict.update(pretrain_dict)
        net_model1.load_state_dict(model_dict)
        net_model1.to(self.device)

        net_model2 = Dynamic_ResNet18().to(self.device)
        pretrain_dict = torch.load(
            './data/net_model2.pth')  # One model, no ensembling
        model_dict = net_model2.state_dict()
        pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict}
        model_dict.update(pretrain_dict)
        net_model2.load_state_dict(model_dict)
        net_model2.to(self.device)

        optimizer_3 = torch.optim.SGD(net_model1.parameters(), lr=0.01)
        optimizer_4 = torch.optim.SGD(net_model2.parameters(), lr=0.01)
        optimizer_5 = torch.optim.SGD(self.autoencoder_2.parameters(), lr=0.01)

        alpha_array = []

        self.a_1 = 0.2
        self.a_2 = 0.8
        self.cl_1 = []
        self.cl_2 = []
        b = -0.08
        "train the network"
        for (i, x) in enumerate(data2):
            x = x.unsqueeze(0).float().to(self.device)

            y = self.label_S2[i].unsqueeze(0).long().to(self.device)

            encoded, decoded = self.autoencoder_2(x)
            optimizer_5.zero_grad()

            loss_4, y_hat_2 = self.HB_Fit(net_model2, encoded, y, optimizer_4)
            loss_3, y_hat_1 = self.HB_Fit(net_model1, encoded, y, optimizer_3)

            loss_5 = self.BCELoss(torch.sigmoid(decoded), x)
            loss_5.backward()
            optimizer_5.step()

            y_hat = self.a_1 * y_hat_1 + self.a_2 * y_hat_2

            self.cl_1.append(loss_3)
            self.cl_2.append(loss_4)

            if len(self.cl_1) == 100:
                self.cl_1.pop(0)
                self.cl_2.pop(0)

            try:
                a_cl_1 = math.exp(b * sum(self.cl_1))
                a_cl_2 = math.exp(b * sum(self.cl_2))
                self.a_1 = (a_cl_1) / (a_cl_2 + a_cl_1)
            except OverflowError:
                self.a_1 = float('inf')

            self.a_2 = 1 - self.a_1

            _, predicted = torch.max(y_hat.data, 1)
            self.correct += (predicted == y).item()

            if i == 0:
                print("finish 1")
            if (i + 1) % 100 == 0:
                print("step : %d" % (i + 1), end=", ")
                print("correct: %d" % (self.correct))
            if (i + 1) % 1000 == 0:
                self.accuracy = self.correct / 1000
                self.S1_lossFigure_i.append(i + 50000)
                self.S1_Figure_Accuracy.append(self.accuracy)
                self.correct = 0
                print("Accuracy: ", self.accuracy)
                print(self.a_1, self.a_2)

        torch.save(self.S1_Figure_Accuracy, './data/old3s_accuracy')

    def zero_grad(self, model):
        for child in model.children():
            for param in child.parameters():
                if param.grad is not None:
                    # param.grad.detach_()
                    param.grad.zero_()  # data.fill_(0)

    def HB_Fit(self, model, X, Y, optimizer, block_split=6):  # hedge backpropagation
        predictions_per_layer = model.forward(X)

        losses_per_layer = []
        for out in predictions_per_layer:
            loss = self.CELoss(out, Y)
            losses_per_layer.append(loss)

        output = torch.empty_like(predictions_per_layer[0])
        for i, out in enumerate(predictions_per_layer):
            # print(self.alpha1[i])
            output += self.alpha1[i] * out

        for i in range(self.num_block):  # First 6 are shallow and last 2 are deep
            # alpha_sum_1 = self.alpha1[0] + self.alpha1[1] + self.alpha1[2] + self.alpha1[3] + self.alpha1[4] + self.alpha1[5]
            if i < block_split:
                if i == 0:
                    alpha_sum_1 = self.alpha1[i]
                else:
                    alpha_sum_1 += self.alpha1[i]
            # alpha_sum_2 = self.alpha1[6] + self.alpha1[7]
            else:
                if i == block_split:
                    alpha_sum_2 = self.alpha1[i]
                else:
                    alpha_sum_2 += self.alpha1[i]

        Loss_sum = torch.zeros_like(losses_per_layer[0])

        for i, loss in enumerate(losses_per_layer):
            if i < block_split:
                loss_ = (self.alpha1[i] / alpha_sum_1) * self.beta1 * loss
            else:
                loss_ = (self.alpha1[i] / alpha_sum_2) * self.beta2 * loss

            Loss_sum += loss_
        optimizer.zero_grad()

        Loss_sum.backward(retain_graph=True)
        optimizer.step()

        for i in range(len(losses_per_layer)):
            self.alpha1[i] *= torch.pow(self.b, losses_per_layer[i])
            self.alpha1[i] = torch.max(self.alpha1[i], self.s / self.num_block)
            self.alpha1[i] = torch.min(self.alpha1[i], self.m)  # exploration-exploitation

        z_t = torch.sum(self.alpha1)
        self.alpha1 = Parameter(self.alpha1 / z_t, requires_grad=False).to(self.device)
        return Loss_sum, output

def loadcifar():

    Newfeature = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(hue=0.3),
        torchvision.transforms.ToTensor()])

    cifar10_original = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )

    cifar10_color = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=Newfeature
    )
    x_S1 = torch.Tensor(cifar10_original.data)
    x_S2 = torch.Tensor(cifar10_color.data)
    y_S1, y_S2 = torch.Tensor(cifar10_original.targets), torch.Tensor(cifar10_color.targets)
    x_S1, y_S1 = shuffle(x_S1, y_S1, random_state=30)
    x_S2, y_S2 = shuffle(x_S2, y_S2, random_state=30)
    x_S1 = torch.transpose(x_S1, 3, 2)
    x_S1 = torch.transpose(x_S1, 2, 1)
    x_S2 = torch.transpose(x_S2, 3, 2)
    x_S2 = torch.transpose(x_S2, 2, 1)

    return x_S1, x_S2, y_S1, y_S2
