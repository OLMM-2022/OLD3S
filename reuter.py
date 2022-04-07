import math
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
from torch.nn.functional import normalize

from torch import nn
from torch.nn.parameter import Parameter
import torch
import torch.nn as nn
import copy
import math
import torch.nn.functional as F



class LoadReuter():
    '''    Instances X Features
        EN_EN: 18758 X 21531 (origin)
        EN_FR: 18758 X 24893
        EN_GR: 18758 X 34215
        EN_IT: 18758 X 15506
        EN_SP: 18758 X 11547
    '''

    def __init__(self, fromLang, dataset_path, seed=42):
        print('LOAD Reuter from: ', str(fromLang))
        self.fromLang = fromLang
        self.dataset_path = dataset_path
        dataOrigin = pd.read_csv(
            self.dataset_path + 'Reuter/' + str(fromLang) + '/Index_' + str(fromLang) + '-' + str(fromLang),
            header=None).values
        self.data, self.label = ReuterPreprocess(dataOrigin, seed=seed)

    def normlize(self, norm='l1'):
        if norm == 'l2':
            self.data = normalize(self.data, norm='l2', axis=0)
        elif norm == 'L1':
            self.data = normalize(self.data, norm='l1', axis=0)
   

    def translate(self, toLang, seed=42):
        fromLang = self.fromLang
        print('LOAD Reuter from: ', str(fromLang), "to", str(toLang))

        dataTrans = pd.read_csv(
            self.dataset_path + 'Reuter/' + str(toLang) + '/Index_' + str(self.fromLang) + '-' + str(toLang),
            header=None).values
        data, label = ReuterPreprocess(dataTrans, seed=seed)
        return data, label

    def enlargerEvo(self, toLang=None, overlapRate=0.1, seed=42):
        print('to: ', str(toLang))
        dataTrans = pd.read_csv(
            self.dataset_path + 'Reuter/' + str(toLang) + '/Index_' + str(self.fromLang) + '-' + str(toLang),
            header=None).values
        dataT, labelT = ReuterPreprocess(dataTrans, seed=seed, startkey=len(self.data))
        overlap_start = math.floor(len(self.data) * (1 - overlapRate))
        for i in range(overlap_start, len(self.data)):
            self.data[i].update(dataT.pop(overlap_start))
            labelT = np.delete(labelT, overlap_start, 0)
        self.data.extend(dataT)
        self.label = np.append(self.label, labelT, axis=0)

class BasicBlock(nn.Module):

    def __init__(self, in_planes, planes):
        super(BasicBlock, self).__init__()
        self.Linear1 = nn.Linear(
            in_planes, planes)
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.Linear1(x)
        out = self.relu(out)
        return out

class MLP(nn.Module):

    def __init__(self, in_planes, num_classes=6):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        super(MLP, self).__init__()
        self.in_planes = in_planes
        self.num_classes = num_classes
        self.Linear = nn.Linear(self.in_planes, 1024)
        self.hidden_layers = []
        self.output_layers = []

        self.hidden_layers.append(BasicBlock(1024, 1024))
        self.hidden_layers.append(BasicBlock(1024, 1024))
        self.hidden_layers.append(BasicBlock(1024, 1024))
        self.hidden_layers.append(BasicBlock(1024, 1024))

        self.output_layers.append(self._make_mlp1(1024))
        self.output_layers.append(self._make_mlp2(1024))
        self.output_layers.append(self._make_mlp3(1024))
        self.output_layers.append(self._make_mlp4(1024))
        self.output_layers.append(self._make_mlp4(1024))

        self.hidden_layers = nn.ModuleList(self.hidden_layers)  #
        self.output_layers = nn.ModuleList(self.output_layers)  #

    def _make_mlp1(self, in_planes):
        classifier = nn.Sequential(

            nn.Linear(in_planes, 6),

        )
        return classifier

    def _make_mlp2(self, in_planes):
        classifier = nn.Sequential(
            nn.Linear(in_planes, 6),

        )
        return classifier

    def _make_mlp3(self, in_planes):
        classifier = nn.Sequential(
            nn.Linear(in_planes, 6),

        )
        return classifier

    def _make_mlp4(self, in_planes):
        classifier = nn.Sequential(
            nn.Linear(in_planes, 6),

        )
        return classifier

    def _make_mlp5(self, in_planes):
        classifier = nn.Sequential(
            nn.Linear(in_planes, 6),

        )
        return classifier

    def forward(self, x):
        hidden_connections = []
        hidden_connections.append(F.relu(self.Linear(x)))

        for i in range(len(self.hidden_layers)):
            hidden_connections.append(self.hidden_layers[i](hidden_connections[i]))

        output_class = []
        for i in range(len(self.output_layers)):
            output = self.output_layers[i](hidden_connections[i])
            output_class.append(output)

        return output_class


def ReuterPreprocess(data, seed=42, startkey=0):
    dataNew = []
    label = []
    for i, row in enumerate(data):
        # print(row)
        string = row[0]
        string = string.split(' ')
        label.append(string.pop(0))
        tempDict = {}
        for _, dict in enumerate(string):
            # print(dict)
            k, p, v = dict.partition(':')
            if p != ':':
                continue
            stringDict = {startkey + int(k): float(v)}

            tempDict.update(stringDict)

        dataNew.append(tempDict)
    y = pd.DataFrame(label)
    y = y.replace('C15', 0)
    y = y.replace('CCAT', 1)
    y = y.replace('E21', 2)
    y = y.replace('ECAT', 3)
    y = y.replace('GCAT', 4)
    y = y.replace('M11', 5)
    y = y.values.reshape(-1, 1)
    data, label = shuffle(dataNew, y, random_state=seed)

    return data, label
    # return dataNew, y


def addZERO(dataset, size):
    data = torch.zeros(size)
    i = 0
    for dict in dataset:
        for j in dict.keys():
            data[i][j - 1] = dict[j]
        i += 1
    return data


class AE(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(AE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(inplanes, outplanes),
            nn.ReLU()

        )
        self.decoder = nn.Sequential(
            nn.Linear(outplanes, inplanes),
            nn.ReLU()
        )

    def forward(self, x):
        encoder = self.encoder(x)
        decoder = self.decoder(encoder)
        return encoder, decoder


class Reuter:
    def __init__(self, data_S1, label_S1, data_S2, label_S2, T1, t, dimension1, dimension2, lr, FromLan,ToLan, b=0.9, s=0.008,
                 m=0.99):

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.correct = 0
        self.accuracy = 0
        self.lr = lr
        self.T1 = T1  # for data come from S1
        self.t = t
        self.B = self.T1 - self.t  # start of an evolving period of time
        self.alpha_loss = [[], [], [], [], []]
        self.x_S1 = data_S1
        self.y_S1 = label_S1
        self.x_S2 = data_S2
        self.y_S2 = label_S2
        self.ToLan = ToLan
        self.FromLan = FromLan
        self.dimension1 = dimension1
        self.dimension2 = dimension2
        self.b = Parameter(torch.tensor(b), requires_grad=False).to(self.device)
        self.s = Parameter(torch.tensor(s), requires_grad=False).to(self.device)
        self.m = Parameter(torch.tensor(m), requires_grad=False).to(self.device)
        self.a_1 = 0.8
        self.a_2 = 0.2
        self.cl_1 = []
        self.cl_2 = []
        self.CELoss = nn.CrossEntropyLoss()
        self.BCELoss = nn.BCELoss()
        self.SmoothL1Loss = nn.SmoothL1Loss()
        self.MSELoss = nn.MSELoss()

        self.S1_lossFigure_i = []
        self.S1_Figure_Accuracy = []

        self.alpha = Parameter(torch.Tensor(5).fill_(1 / 5), requires_grad=False).to(
            self.device)

        self.autoencoder_1 = AE(self.dimension1, 1024).to(self.device)
        self.autoencoder_2 = AE(self.dimension2, 1024).to(self.device)

    def T_1(self):
        loss_1_array = []
        classifier_1 = MLP(1024).to(self.device)
        optimizer_classifier_1 = torch.optim.SGD(classifier_1.parameters(), self.lr)
        optimizer_autoencoder_1 = torch.optim.SGD(self.autoencoder_1.parameters(), self.lr)
        b = -0.01

        for (i, x) in enumerate(self.x_S1):

            self.i = i
            x1 = x.unsqueeze(0).float().to(self.device)
            y = self.y_S1[i].long().to(self.device)

            '''train Autoencoder'''
            if self.i < self.B:  # Before evolve
                encoded_1, decoded_1 = self.autoencoder_1(x1)
                optimizer_autoencoder_1.zero_grad()
                y_hat, loss_classifier_1 = self.HB_Fit(classifier_1,
                                                       encoded_1, y, optimizer_classifier_1)

                loss_autoencoder_1 = self.BCELoss(torch.sigmoid(decoded_1), x1)
                loss_autoencoder_1.backward()
                optimizer_autoencoder_1.step()

            else:  # Evolving start
                x2 = self.x_S2[self.i].unsqueeze(0).float().to(self.device)
                if i == self.B:
                    classifier_2 = copy.deepcopy(classifier_1)

                    torch.save(classifier_1.state_dict(),
                               './data/'+self.FromLan + '_' + self.ToLan + '/net_model1.pth')
                    optimizer_classifier_2 = torch.optim.SGD(classifier_2.parameters(), self.lr)
                    optimizer_autoencoder_2 = torch.optim.SGD(self.autoencoder_2.parameters(), 0.99)

                encoded_2, decoded_2 = self.autoencoder_2(x2)
                encoded_1, decoded_1 = self.autoencoder_1(x1)

                y_hat_2, loss_classifier_2 = self.HB_Fit(classifier_2,
                                                         encoded_2, y, optimizer_classifier_2)
                y_hat_1, loss_classifier_1 = self.HB_Fit(classifier_1,
                                                         encoded_1, y, optimizer_classifier_1)

                y_hat = self.a_1 * y_hat_1 + self.a_2 * y_hat_2

                self.cl_1.append(loss_classifier_1)
                self.cl_2.append(loss_classifier_2)
                if len(self.cl_1) == 50:
                    self.cl_1.pop(0)
                    self.cl_2.pop(0)

                try:
                    a_cl_1 = math.exp(b * sum(self.cl_1))
                    a_cl_2 = math.exp(b * sum(self.cl_2))
                    self.a_1 = (a_cl_1) / (a_cl_2 + a_cl_1)

                except OverflowError:
                    self.a_1 = float('inf')

                self.a_2 = 1 - self.a_1

                optimizer_autoencoder_2.zero_grad()
                loss_2_0 = self.BCELoss(torch.sigmoid(decoded_2), x2)
                loss_2_1 = self.SmoothL1Loss(encoded_2, encoded_1)
                loss_autoencoder_2 = loss_2_0 + loss_2_1
                loss_autoencoder_2.backward()
                optimizer_autoencoder_2.step()

            loss_1_array.append(loss_classifier_1.item())

            _, predicted = torch.max(y_hat.data, 1)

            self.correct += (predicted == y).item()
            if i == 0:
                print("finish 0")
            if (i + 1) % 100 == 0:
                print("step : %d" % (i + 1), end=", ")
                print("correct: %d" % (self.correct))

            if (i + 1) % 500 == 0:
                self.accuracy = self.correct / 500
                self.S1_lossFigure_i.append(i)
                self.S1_Figure_Accuracy.append(self.accuracy)
                self.correct = 0
                print("Accuracy: ", self.accuracy)

        torch.save(classifier_2.state_dict(),
                   './data/'+self.FromLan + '_' + self.ToLan + '/net_model2.pth')
        print('model2 load successfully')

    def T_2(self):
      
        self.T_1()

        self.correct = 0

        "load data for T2"
        data2 = self.x_S2[:self.B]
        "use the Network"
        net_model1 = MLP(1024).to(self.device)
        pretrain_dict = torch.load(
            './data/'+self.FromLan + '_' + self.ToLan + '/net_model1.pth')  # One model, no ensembling
        model_dict = net_model1.state_dict()
        pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict}
        model_dict.update(pretrain_dict)
        net_model1.load_state_dict(model_dict)
        net_model1.to(self.device)

        net_model2 = MLP(1024).to(self.device)
        pretrain_dict = torch.load(
            './data/'+self.FromLan + '_' + self.ToLan + '/net_model2.pth')  # One model, no ensembling
        model_dict = net_model2.state_dict()
        pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict}
        model_dict.update(pretrain_dict)
        net_model2.load_state_dict(model_dict)
        net_model2.to(self.device)

        "train the network"
        optimizer_classifier_1_FES = torch.optim.SGD(net_model1.parameters(), self.lr)
        optimizer_classifier_2_FES = torch.optim.SGD(net_model2.parameters(), self.lr)
        optimizer_autoencoder_2_FES = torch.optim.SGD(self.autoencoder_2.parameters(), self.lr)
        data_2 = self.x_S2[:self.B]
        label_2 = self.y_S1[:self.B]

        self.a_1 = 0.3
        self.a_2 = 0.7
        self.cl_1 = []
        self.cl_2 = []
        b = -0.01
        for (i, x) in enumerate(data_2):
            x = x.unsqueeze(0).float().to(self.device)
            self.i = i + self.T1
            y = label_2[i].long().to(self.device)

            encoded_2, decoded_2 = self.autoencoder_2(x)
            optimizer_autoencoder_2_FES.zero_grad()
            y_hat_2, loss_classifier_2 = self.HB_Fit(net_model2,
                                                     encoded_2, y, optimizer_classifier_2_FES)
            y_hat_1, loss_classifier_1 = self.HB_Fit(net_model1,
                                                     encoded_2, y, optimizer_classifier_1_FES)

            loss_autoencoder_2 = self.BCELoss(torch.sigmoid(decoded_2), x)
            loss_autoencoder_2.backward()
            optimizer_autoencoder_2_FES.step()

            y_hat = self.a_1 * y_hat_1 + self.a_2 * y_hat_2
            '''self.a_1 * y_hat_1 + self.a_2 *'''
            self.cl_1.append(loss_classifier_1)
            self.cl_2.append(loss_classifier_2)

            if len(self.cl_1) == 50:
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

            if (i + 1) % 500 == 0:
                self.accuracy = self.correct / 500
                self.S1_lossFigure_i.append(i + 26500)
                self.S1_Figure_Accuracy.append(self.accuracy)
                self.correct = 0

                print("Accuracy: ", self.accuracy)

        torch.save(self.S1_Figure_Accuracy, './data/'+self.FromLan + '_' + self.ToLan + '/reuter_accuracy')

    def zero_grad(self, model):
        for child in model.children():
            for param in child.parameters():
                if param.grad is not None:
                    # param.grad.detach_()
                    param.grad.zero_()  # data.fill_(0)

    def isEqual(self, predict, label):
        predictor = predict.argmax()

        real = label.argmax()
        return predictor == real

    def HB_Fit(self, model, X, Y, optimizer):  # hedge backpropagation
        predictions_per_layer = model.forward(X)

        losses_per_layer = []
        for out in predictions_per_layer:
            loss = self.CELoss(out, Y)
            losses_per_layer.append(loss)

        output = torch.empty_like(predictions_per_layer[0])
        for i, out in enumerate(predictions_per_layer):
            output += self.alpha[i] * out

        for i in range(5):  # First 6 are shallow and last 2 are deep

            if i == 0:
                alpha_sum_1 = self.alpha[i]
            else:
                alpha_sum_1 += self.alpha[i]

        Loss_sum = torch.zeros_like(losses_per_layer[0])

        for i, loss in enumerate(losses_per_layer):
            loss_ = (self.alpha[i] / alpha_sum_1) * loss
            Loss_sum += loss_
        optimizer.zero_grad()

        Loss_sum.backward(retain_graph=True)
        optimizer.step()

        for i in range(len(losses_per_layer)):
            self.alpha[i] *= torch.pow(self.b, losses_per_layer[i])
            self.alpha[i] = torch.max(self.alpha[i], self.s / 5)
            self.alpha[i] = torch.min(self.alpha[i], self.m)  # exploration-exploitation

        z_t = torch.sum(self.alpha)
        self.alpha = Parameter(self.alpha / z_t, requires_grad=False).to(self.device)

        return output, Loss_sum





def SaveDataSets(from_dataset, to_dataset, samples, size1, size2):
    dataset_path = './data/'
    FRdataset = LoadReuter(fromLang=from_dataset, dataset_path=dataset_path)
    FRdata, FRlabel = FRdataset.data, FRdataset.label
    Todata, Tolabel = FRdataset.translate(toLang=to_dataset)
    x_S1 = addZERO(FRdata, (samples, size1))
    x_S2 = addZERO(Todata, (samples, size2))
    y_S1, y_S2 = torch.Tensor(FRlabel), torch.Tensor(Tolabel)
    
    if from_dataset == 'FR':
        pca_from = PCA(n_components=2500)
    else:
        pca_from = PCA(n_components=2000)
    
    if to_dataset == 'FR':
        pca_to = PCA(n_components=2500)
    elif to_dataset == 'SP':
        pca_to = PCA(n_components=1000)
    else:
        pca_to = PCA(n_components=1500)
    newx1 = pca_from.fit_transform(x_S1)
    newx2 = pca_to.fit_transform(x_S2)
    
    torch.save(newx1,'./data/'+ from_dataset + '_'+to_dataset +'/x_S1_pca')
    torch.save(newx2,'./data/'+ from_dataset + '_'+to_dataset +'/x_S2_pca')
    torch.save(y_S1, './data/' + from_dataset + '_' + to_dataset + '/y_S1')
    torch.save(y_S2, './data/' + from_dataset + '_' + to_dataset + '/y_S2')


def loadreuter(from_dataset, to_dataset, samples, size1, size2):
    print('start to generate data and labels')
    SaveDataSets(from_dataset, to_dataset, samples, size1, size2)
    x_S1 = torch.load('./data/' + from_dataset + '_' + to_dataset + '/x_S1_pca')
    x_S2 = torch.load('./data/' + from_dataset + '_' + to_dataset + '/x_S2_pca')
    y_S1 = torch.load('./data/' + from_dataset + '_' + to_dataset + '/y_S1')
    y_S2 = torch.load('./data/' + from_dataset + '_' + to_dataset + '/y_S2')

    return x_S1, x_S2, y_S1, y_S2
