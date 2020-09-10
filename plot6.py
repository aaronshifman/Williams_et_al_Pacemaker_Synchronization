import os
import numpy as np
import pandas as pd
import pickle as pic

from sklearn.model_selection import train_test_split
from scipy.optimize import curve_fit

import torch as tc
from torch import nn
from torch import optim


def inv_function(array, c):
    """
    element-wise inverse function c/array
    :param array: input data
    :param c: parameter
    :return: c/array
    """

    return c/array


def z_score(array, return_stats=False):
    """
    Z-scores an input array of data. Assumes that mean/sd are to be taken over first dimension
    :param array: input, 2D array
    :param return_stats: bool; whether to give back mean, sd
    :return: x but z-scored
    """

    m = np.mean(array, axis=0)
    sd = np.std(array, axis=0)

    if return_stats:
        return (array - m) / sd, m, sd
    else:
        return (array - m) / sd


def robust_sync(df):
    """
    function adds a column to dataframe to denote which graph types exhibit robust synchronization
    :param data:
    :return:
    """

    not_sync = df['SyncTime'].values == -1
    ns_list = np.split(not_sync, 4000)
    robust_sync = np.asarray([np.sum(val) <= 1 for val in ns_list])

    df['Robust'] = robust_sync.repeat(20)

    return df


def get_type_ind(df_valid):
    """
    gets type_index from dataframe
    :param df_valid: dataframe
    :return: type_index
    """
    uc_ind6 = (df_valid.Direction.values == 'undirected') & (df_valid.Randomness.values == 'constant') & (df_valid.k.values == 6)
    ur_ind6 = (df_valid.Direction.values == 'undirected') & (df_valid.Randomness.values == 'random') & (df_valid.k.values == 6)
    dr_ind6 = (df_valid.Direction.values == 'directed') & (df_valid.Randomness.values == 'random') & (df_valid.k.values == 6)
    dci_ind6 = (df_valid.Direction.values == 'directed') & (df_valid.Randomness.values == 'constant') & (df_valid.k.values == 6)
    dco_ind6 = (df_valid.Direction.values == 'directed') & (df_valid.Randomness.values == 'constant_out') & (df_valid.k.values == 6)
    dfull_ind6 = (df_valid.Direction.values == 'directed') & (df_valid.Randomness.values == 'full') & (df_valid.k.values == 6)

    uc_ind4 = (df_valid.Direction.values == 'undirected') & (df_valid.Randomness.values == 'constant') & (df_valid.k.values == 4)
    ur_ind4 = (df_valid.Direction.values == 'undirected') & (df_valid.Randomness.values == 'random') & (df_valid.k.values == 4)
    dr_ind4 = (df_valid.Direction.values == 'directed') & (df_valid.Randomness.values == 'random') & (df_valid.k.values == 4)
    dci_ind4 = (df_valid.Direction.values == 'directed') & (df_valid.Randomness.values == 'constant') & (df_valid.k.values == 4)
    dco_ind4 = (df_valid.Direction.values == 'directed') & (df_valid.Randomness.values == 'constant_out') & (df_valid.k.values == 4)
    dfull_ind4 = (df_valid.Direction.values == 'directed') & (df_valid.Randomness.values == 'full') & (df_valid.k.values == 4)

    uc_ind = uc_ind4 * 1
    ur_ind = ur_ind4 * 1
    dr_ind = dr_ind4 * 1
    dci_ind = dci_ind4 * 1
    dco_ind = dco_ind4 * 1
    dfull_ind = dfull_ind4 * 1
    q_ind = df_valid.q.values
    uc_ind6 = uc_ind6 * 1
    ur_ind6 = ur_ind6 * 1
    dr_ind6 = dr_ind6 * 1
    dci_ind6 = dci_ind6 * 1
    dco_ind6 = dco_ind6 * 1
    dfull_ind6 = dfull_ind6 * 1

    return np.concatenate((np.expand_dims(ur_ind, axis=1), np.expand_dims(uc_ind, axis=1),
                                 np.expand_dims(dr_ind, axis=1), np.expand_dims(dci_ind, axis=1),
                                 np.expand_dims(dco_ind, axis=1), np.expand_dims(dfull_ind, axis=1),
                                 np.expand_dims(q_ind, axis=1), np.expand_dims(ur_ind6, axis=1),
                                 np.expand_dims(uc_ind6, axis=1), np.expand_dims(dr_ind6, axis=1),
                                 np.expand_dims(dci_ind6, axis=1), np.expand_dims(dco_ind6, axis=1),
                                 np.expand_dims(dfull_ind6, axis=1)), axis=1)


def get_data():
    """
    loads dataframe from csv file and gets data for model fitting and plotting
    :return:
    """

    # Import------------------------------------------------------------------------------------------------------------
    cwd = os.getcwd() + '/'
    file = 'ML.csv'

    converters = {'e' + str(ix): lambda x: np.complex(x) if x != 'nan' else np.nan for ix in np.arange(100)}
    data = pd.read_csv(cwd + file, index_col=0, converters=converters)

    # Setup data--------------------------------------------------------------------------------------------------------

    # Remove rows that don't synchronize ROBUSTLY
    data = robust_sync(data)
    drop_index = data.index[np.invert(data['Robust'])]
    data = data.drop(drop_index)
    data = data.reset_index()

    # Remove few networks from robustly synchronizing networks that don't sync
    drop_index = data.index[data['SyncTime'].values == -1]
    data = data.drop(drop_index)
    data = data.reset_index()

    # Set-up basic params for data shape and training / test / validation split
    n_eig = 100
    n_trials = data.shape[0]
    p_train = 0.8
    p_test = 0.1

    # Get all eigs
    x = np.zeros((n_trials, n_eig - 1), dtype='complex')
    for ix in range(1, n_eig):
        x[:, ix - 1] = data['e' + str(ix)]
    y = data['SyncTime'].values

    # Drop eigenvalues beyond smallest set
    last_e = np.min(np.where(np.isnan(x))[1]) - 1
    x = x[:, :last_e]

    # Organize data in 'ML_arrays' file if it hasn't already been done
    if os.path.isdir('ML_arrays_r') is False:

        print('making new ML_arrays folder')
        # Split data
        df_train, df_ = train_test_split(data, train_size=p_train)
        p_ = p_test / (1 - p_train)
        df_test, df_valid, = train_test_split(df_,train_size=p_)

        x_train = x[df_train.index]
        y_train = df_train['SyncTime'].values
        x_test = x[df_test.index]
        y_test = df_test['SyncTime'].values
        x_valid = x[df_valid.index]
        y_valid = df_valid['SyncTime'].values

        os.mkdir('ML_arrays_r')
        np.save('ML_arrays_r/y', y)
        np.save('ML_arrays_r/x', x)
        np.save('ML_arrays_r/x_train', x_train)
        np.save('ML_arrays_r/y_train', y_train)
        np.save('ML_arrays_r/x_test', x_test)
        np.save('ML_arrays_r/y_test', y_test)
        np.save('ML_arrays_r/x_valid', x_valid)
        np.save('ML_arrays_r/y_valid', y_valid)
        with open('ML_arrays_r/df_train', 'wb') as f_save:
            pic.dump(df_train, f_save)
        with open('ML_arrays_r/df_test', 'wb') as f_save:
            pic.dump(df_test, f_save)
        with open('ML_arrays_r/df_valid', 'wb') as f_save:
            pic.dump(df_valid, f_save)

    else:

        print('Loading data...')
        # y = np.load('ML_arrays_r/y.npy')
        # x = np.load('ML_arrays_r/x.npy')
        x_train = np.load('ML_arrays_r/x_train.npy')
        y_train = np.load('ML_arrays_r/y_train.npy')
        x_test = np.load('ML_arrays_r/x_test.npy')
        y_test = np.load('ML_arrays_r/y_test.npy')
        x_valid = np.load('ML_arrays_r/x_valid.npy')
        y_valid = np.load('ML_arrays_r/y_valid.npy')
        with open('ML_arrays_r/df_train', 'rb') as load_f:  # Use if file size is small enough
            df_train = pic.load(load_f)
        with open('ML_arrays_r/df_test', 'rb') as load_f:  # Use if file size is small enough
            df_test = pic.load(load_f)
        with open('ML_arrays_r/df_valid', 'rb') as load_f:  # Use if file size is small enough
            df_valid = pic.load(load_f)

    # Get indices for degree = 4, 6
    type_ind_train = get_type_ind(df_train)
    type_ind_test = get_type_ind(df_test)
    type_ind_valid = get_type_ind(df_valid)
    itr4 = np.any(type_ind_train[:, 0:6] == 1, axis=1)
    ite4 = np.any(type_ind_test[:, 0:6] == 1, axis=1)
    iva4 = np.any(type_ind_valid[:, 0:6] == 1, axis=1)
    itr6 = np.any(type_ind_train[:, 7:12] == 1, axis=1)
    ite6 = np.any(type_ind_test[:, 7:12] == 1, axis=1)
    iva6 = np.any(type_ind_valid[:, 7:12] == 1, axis=1)

    xtr4 = np.real(x_train[itr4, :]).astype(np.float64)
    xte4 = np.real(x_test[ite4, :]).astype(np.float64)
    xva4 = np.real(x_valid[iva4, :]).astype(np.float64)
    ytr4 = y_train[itr4].astype(np.float64)
    yte4 = y_test[ite4].astype(np.float64)
    yva4 = y_valid[iva4].astype(np.float64)

    xtr6 = np.real(x_train[itr6, :]).astype(np.float64)
    xte6 = np.real(x_test[ite6, :]).astype(np.float64)
    xva6 = np.real(x_valid[iva6, :]).astype(np.float64)
    ytr6 = y_train[itr6].astype(np.float64)
    yte6 = y_test[ite6].astype(np.float64)
    yva6 = y_valid[iva6].astype(np.float64)

    X4 = (xtr4, xte4, xva4)
    Y4 = (ytr4, yte4, yva4)
    X6 = (xtr6, xte6, xva6)
    Y6 = (ytr6, yte6, yva6)

    return X4, Y4, X6, Y6, type_ind_valid, data.q.unique(), np.real(x_valid[:, 0]), y_valid


def MSE_loss_reg(output, target, weights=None, L1=None, L2=None):
    """
    updates MSE_loss with L1 and L2 loss
    :param output:
    :param target:
    :param weights:
    :param L1:
    :param L2:
    :return:
    """

    loss_fn = nn.MSELoss()
    loss = loss_fn(output, target)

    if weights is not None and L2 is not None:
        loss += L1 * tc.absolute(weights).sum() + L2 * tc.square(weights).sum()

    return loss


class FeedForward(nn.Module):
    """
    Feed-forward ANN with one hidden layer and one output neuron (nonlinear regression)
    """

    def __init__(self, n_inputs, n_hidden):
        super().__init__()
        self.mat1 = nn.Linear(n_inputs, n_hidden).double()
        self.mat2 = nn.Linear(n_hidden, 1).double()

    def forward(self, x):

        h = tc.tanh(self.mat1(x))
        y = self.mat2(h)

        return y

    def my_train(self, train_data, train_labels, test_data=None, test_labels=None, n_epochs=5, n_batch=64, learning_rate=0.001, momentum=0.99, L1=None, L2=None):

        n_train = train_data.shape[0]
        #optimizer = optim.SGD(self.parameters(), lr=learning_rate, momentum=momentum)
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        test_loss = []
        train_loss = []

        print('Training net...')
        for epoch in range(n_epochs):

            batch_index = tc.randperm(n_train)
            batch_data = tc.split(train_data[batch_index], n_batch)
            batch_labels = tc.split(train_labels[batch_index], n_batch)

            for val in zip(batch_data, batch_labels):

                out = self.forward(val[0])
                loss = MSE_loss_reg(out, val[1], weights=self.mat1.weight, L1=L1, L2=L2)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            train_loss.append(MSE_loss_reg(self.forward(train_data), train_labels).item())
            test_loss.append(MSE_loss_reg(self.forward(test_data), test_labels).item())
            print(f'Epoch: {epoch + 1}; Train Loss: {train_loss[epoch]}; Test Loss: {test_loss[epoch]}')

        return test_loss, train_loss


def set_up_lists(X4, Y4, X6, Y6):
    """
    organizes, z-scores and converts to torch tensors for ANN training
    :param X4:
    :param Y4:
    :param X6:
    :param Y6:
    :return:
    """

    X_train_list = [tc.from_numpy(z_score(X4[0][:, :1])), tc.from_numpy(z_score(X4[0][:, :20])),
                    tc.from_numpy(z_score(X4[0][:, :90])), tc.from_numpy(z_score(X6[0][:, :1])),
                    tc.from_numpy(z_score(X6[0][:, :20])), tc.from_numpy(z_score(X6[0][:, :90]))]
    X_test_list = [tc.from_numpy(z_score(X4[1][:, :1])), tc.from_numpy(z_score(X4[1][:, :20])),
                   tc.from_numpy(z_score(X4[1][:, :90])), tc.from_numpy(z_score(X6[1][:, :1])),
                   tc.from_numpy(z_score(X6[1][:, :20])), tc.from_numpy(z_score(X6[1][:, :90]))]
    X_valid_list = [tc.from_numpy(z_score(X4[2][:, :1])), tc.from_numpy(z_score(X4[2][:, :20])),
                   tc.from_numpy(z_score(X4[2][:, :90])), tc.from_numpy(z_score(X6[2][:, :1])),
                   tc.from_numpy(z_score(X6[2][:, :20])), tc.from_numpy(z_score(X6[2][:, :90]))]

    Y_train_list = [tc.from_numpy(Y4[0][:, np.newaxis]), tc.from_numpy(Y4[0][:, np.newaxis]),
                    tc.from_numpy(Y4[0][:, np.newaxis]), tc.from_numpy(Y6[0][:, np.newaxis]),
                    tc.from_numpy(Y6[0][:, np.newaxis]), tc.from_numpy(Y6[0][:, np.newaxis])]
    Y_test_list = [tc.from_numpy(Y4[1][:, np.newaxis]), tc.from_numpy(Y4[1][:, np.newaxis]),
                   tc.from_numpy(Y4[1][:, np.newaxis]), tc.from_numpy(Y6[1][:, np.newaxis]),
                   tc.from_numpy(Y6[1][:, np.newaxis]), tc.from_numpy(Y6[1][:, np.newaxis])]
    Y_valid_list = [tc.from_numpy(Y4[2][:, np.newaxis]), tc.from_numpy(Y4[2][:, np.newaxis]),
                   tc.from_numpy(Y4[2][:, np.newaxis]), tc.from_numpy(Y6[2][:, np.newaxis]),
                   tc.from_numpy(Y6[2][:, np.newaxis]), tc.from_numpy(Y6[2][:, np.newaxis])]

    X4_curve = np.concatenate((X4[0], X4[1]), axis=0)[:, 0]
    X6_curve = np.concatenate((X6[0], X6[1]), axis=0)[:, 0]
    Y4_curve = np.concatenate((Y4[0], Y4[1]), axis=0)
    Y6_curve = np.concatenate((Y6[0], Y6[1]), axis=0)

    return zip(X_train_list, Y_train_list, X_test_list, Y_test_list, X_valid_list, Y_valid_list), (X4_curve, Y4_curve, X6_curve, Y6_curve)


def RMSE(model, X, Y):
    """
    calculates RMSE for model
    :param model:
    :param X:
    :param Y:
    :return:
    """

    Yhat = model.forward(X).detach().numpy()
    Y = Y.detach().numpy()

    return np.sqrt(np.mean((Yhat - Y) ** 2)) * 1000


def get_axes(model, X, m, sd):
    """
    gets fitted 1-d line to plot (for ANN)
    :param model:
    :param X:
    :param m:
    :param sd:
    :return:
    """

    x = X[:, 0] * sd + m
    x_axis = np.arange(np.min(x), np.max(x), 0.001)
    yhat = model.forward(tc.from_numpy((x_axis[:, np.newaxis] - m) / sd)).detach().numpy()

    return x_axis, yhat


def RMSE_curve(c, X, Y):
    """
    RMSE for simple inverse function
    :param c:
    :param X:
    :param Y:
    :return:
    """

    return np.sqrt(np.mean((Y - c / X)**2)) * 1000


def get_axes_curve(c, X):
    """
    Gets fitted line to plot (for inverse function)
    :param c:
    :param X:
    :return:
    """

    _, mtr, sdtr = z_score(X[:, 0], return_stats=True)
    x_axis = np.arange(np.min(X[:, 0]), np.max(X[:, 0]), 0.001)

    return c / x_axis


def fit_all_models(X4, X6, MEAN, SD, DATA, SEED):
    """
    Messy function that loops through all models and fits them. Parameters of NN are hardcoded in this function
    :param X4:
    :param X6:
    :param MEAN:
    :param SD:
    :param DATA:
    :param SEED
    :return:
    """

    # Neural net params
    #n_inputs = 90
    n_hidden = 100
    epochs = 60
    batch_size = 16 #32 for K4, 16 for K6 #4
    lr = 0.000002
    m = 0 #0.99
    l2 = 0.05 #0.05
    l1 = 0 #0.05

    # predefine lists for saving stuff
    train_stats = []
    test_stats = []

    error_list = []

    y_axes_list = []
    x_axes_list = []

    idx = 0
    n_input_list = np.tile(np.array([1, 20, 90]), 2)

    # Fit ANN models
    for val1, val2, val3, val4, val5, val6 in DATA:

        n_inputs = n_input_list[idx]

        np.random.seed(SEED)
        tc.manual_seed(SEED)

        net = FeedForward(n_inputs, n_hidden)
        test_error, train_error = net.my_train(val1, val2, val3, val4, n_epochs=epochs, n_batch=batch_size, learning_rate=lr, momentum=m, L1=l1, L2=l2)
        train_stats.append(train_error)
        test_stats.append(test_error)

        error_list.append(RMSE(net, val5, val6))

        if (idx == 0) | (idx == 3):
            x_axis, y_axis = get_axes(net, val5.detach().numpy(), MEAN[idx], SD[idx])
            x_axes_list.append(x_axis)
            y_axes_list.append(y_axis)

        idx += 1

    c4, _ = curve_fit(inv_function, DATA_c[0], DATA_c[1])
    c6, _ = curve_fit(inv_function, DATA_c[2], DATA_c[3])

    error_curve = [RMSE_curve(c4, X4[2][:, 0], Y4[2]), RMSE_curve(c6, X6[2][:, 0], Y6[2])]

    y_axes_curve = [get_axes_curve(c4, X4[2]), get_axes_curve(c6, X6[2])]

    return train_stats, test_stats, error_list, x_axes_list, y_axes_list, error_curve, y_axes_curve


def type_scatter(x_axis, y_axis, type_index, index):
    """
    Formats x and y axes for eigenvalue plots when colour is related to type
    :param x_axis:
    :param y_axis:
    :param type_index:
    :param index:
    :return:
    """

    x_save = []
    y_save = []

    for i, ind, in enumerate(index):
        x_save.append(x_axis[type_index[:, ind] == 1])
        y_save.append(y_axis[type_index[:, ind] == 1])

    return y_save, x_save


def q_scatter(x_axis, y_axis, type_index, index, index2):
    """
    Formats x and y axes for eigenvalue plots when colour is related to q-value
    :param x_axis:
    :param y_axis:
    :param type_index:
    :param index:
    :param index2:
    :return:
    """

    x_save = []
    y_save = []

    for i, val, in enumerate(index):
        x_save0 = []
        y_save0 = []

        for _, val2 in enumerate(index2):
            x_save0.extend(x_axis[(type_index[:, 6] == val) & (type_index[:, val2] == 1)])
            y_save0.extend(y_axis[(type_index[:, 6] == val) & (type_index[:, val2] == 1)])

        x_save.append(x_save0)
        y_save.append(y_save0)

    return y_save, x_save


if __name__ == "__main__":

    saving = False

    # Get and organize data
    X4, Y4, X6, Y6, TIV, Q, x_val, y_val = get_data()
    DATA, DATA_c = set_up_lists(X4, Y4, X6, Y6)

    _, m1, sd1 = z_score(X4[2][:, :1], return_stats=True)
    _, m2, sd2 = z_score(X6[2][:, :1], return_stats=True)
    MEAN, SD = np.repeat(np.array([m1, m2]), 3), np.repeat(np.array([sd1, sd2]), 3)

    # Fit models
    train_stats, test_stats, error, x_axes_list, y_axes_list, error_curve, y_axes_curve = \
        fit_all_models(X4, X6, MEAN, SD, DATA, SEED=42)

    # Save everything for plots in dictionary
    dict_save = {}

    # Save stats
    dict_save['training_stats'] = (train_stats, test_stats)

    # Save error
    dict_save['error'] = [error_curve, [error[0], error[3]], [error[1], error[4]], [error[2], error[5]]]

    # Save lines
    dict_save['12_dashline_x'] = x_axes_list[0]
    dict_save['12_dashline_y'] = y_axes_list[0]
    dict_save['12_solidline_x'] = x_axes_list[0]
    dict_save['12_solidline_y'] = y_axes_curve[0]
    dict_save['45_dashline_x'] = x_axes_list[1]
    dict_save['45_dashline_y'] = y_axes_list[1]
    dict_save['45_solidline_x'] = x_axes_list[1]
    dict_save['45_solidline_y'] = y_axes_curve[1]

    # Save scatterplots
    INDEX4T = [0, 1, 2, 3, 4, 5]
    INDEX6T = [7, 8, 9, 10, 11, 12]
    INDEX4Q = Q
    INDEX6Q = Q

    dict_save['1_scatter_y'], dict_save['1_scatter_x'] = type_scatter(x_val, y_val, TIV, INDEX4T)
    dict_save['4_scatter_y'], dict_save['4_scatter_x'] = type_scatter(x_val, y_val, TIV, INDEX6T)
    dict_save['2_scatter_y'], dict_save['2_scatter_x'] = q_scatter(x_val, y_val, TIV, Q, INDEX4T)
    dict_save['5_scatter_y'], dict_save['5_scatter_x'] = q_scatter(x_val, y_val, TIV, Q, INDEX6T)

    if saving:
        with open(os.getcwd() + '/figure6.pkl', 'wb') as f_save:
            pic.dump(dict_save, f_save)













