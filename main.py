from cifar import loadcifar, OLDS
from reuter import loadreuter, OLDS_reuter

if __name__ == '__main__':
    print('trainning starts')
    x_S1, x_S2, y_S1, y_S2 = loadcifar()
    train = OLDS(x_S1, y_S1, x_S2, y_S2, 50000, 5000)
    train.T_2()


