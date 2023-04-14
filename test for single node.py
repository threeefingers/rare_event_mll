#simulation
# %% ---------------set hyper parameters and libraries--------------
#library
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns

# data generation
N_PATIENTS = 10000
N_FEATURES = 10
RANDOM_SEED = 2023
STEP_SIZE = 1e-3

event_rate = 0.01
similarity = 1

# modeling
training_ratio = 0.6
test_ratio = 0.2

learning_rate = 0.001
batch_size = 16

num_layers = 1
hidden_size = 1
epochs = 100

lam = 0e-5

plot = True

# %% ---------------data simulation--------------
# %% def data generation
def generate_data(event_rate, similarity):

	# rs = np.random.RandomState(RANDOM_SEED)
    rs = np.random.RandomState()
        
	# generate N_FEATURES-dimensional feature vector for N_PATIENTS
    x = rs.randn(N_PATIENTS, N_FEATURES).astype(np.float32)

	# generate coefficient vectors for events 1 and 2
    u1, u2 = generate_vectors_by_similarity(rs, N_FEATURES, similarity)

	# find logit offset that gives the desired event rate
    offset = find_offset(
		rs,
		np.dot(x, normed_uniform(rs, N_FEATURES)),
		event_rate,
		STEP_SIZE
	)

	# calculate logits for each event
    l1 = np.dot(x, u1) - offset
    l2 = np.dot(x, u2) - offset

    # calculate probability of each event
    p1 = sigmoid(l1)
    p2 = sigmoid(l2)

    # generate events
    e1 = bernoulli_draw(rs, p1).astype(np.float32)
    e2 = bernoulli_draw(rs, p2).astype(np.float32)
        
    # plot generated data
    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(10, 4))

    ax[0].hist(l1, alpha=.5, bins=20, label='Event 1')
    ax[0].hist(l2, alpha=.5, bins=20, label='Event 2')
    ax[0].set_title('Event Logits')
    ax[0].legend()

    ax[1].hist(p1, alpha=.5, bins=20, label='Event 1')
    ax[1].hist(p2, alpha=.5, bins=20, label='Event 2')
    ax[1].set_title('Event Probabilities')
    ax[1].legend()

    plt.show()

    # print event rate and similarity
    print(event_rate)

    return x, e1, e2


def generate_vectors_by_similarity(rs, n, s):

	# generate vector 1
	u1 = normed_uniform(rs, n)

	# generate a vector orthogonal to v1
	u1_ = normed_uniform(rs, n)
	u1_ = normalize(u1_ - u1 * np.dot(u1, u1_))

	# generate vector 2
	u2 = u1 * s + u1_ * (1 - s)

	return u1, u2


def find_offset(rs, logits, event_rate, step_size):

	offset = 0.
	rate = 1.

	while rate > event_rate:

		offset += step_size
		p = sigmoid(logits - offset)
		rate = np.mean(bernoulli_draw(rs, p))

	return offset


def normed_uniform(rs, n):
	return normalize(rs.rand(n) - .5)


def bernoulli_draw(rs, p):
	return (rs.rand(len(p)) < p).astype(int)


def glorot_uniform(rs, num_in, num_out):
	scale_factor = 2 * np.sqrt(6 / (num_in + num_out))
	return scale_factor * np.squeeze(rs.rand(num_in, num_out) - .5)


def logit(p):
	return np.log(p / (1 - p))


def sigmoid(l):
	return 1 / (1 + np.exp(-1 * l))


def normalize(v):
	return v / np.linalg.norm(v)
	



# %% ---------------single-label model--------------
# %% def data loader

def split_set(training_ratio, test_ratio, data):
    train = data[:int(training_ratio*len(data))]
    valid = data[int(training_ratio*len(data)):int(1-test_ratio*len(data))]
    test = data[int(1-test_ratio*len(data)):]
    return train, valid, test

# %% def data loader
def load_data(data_x, data_y1, data_y2):
    class TransData_m():
        def __init__(self, xx, yy):
            self.X = xx
            self.y = yy

        def __len__(self):
            return self.X.shape[0]

        def __getitem__(self, idx):
            return self.X[idx, :], self.y[idx, :]
    
    data_y = np.vstack((data_y1, data_y2)).T

    x_train, x_valid, x_test = split_set(training_ratio, test_ratio, data_x)
    y_train, y_valid, y_test = split_set(training_ratio, test_ratio, data_y)
    
    train_set = TransData_m(xx=x_train, yy=y_train)
    valid_set = TransData_m(xx=x_valid, yy=y_valid)
    test_set = TransData_m(xx=x_test, yy=y_test)

    train = DataLoader(train_set, batch_size=batch_size, shuffle=True) # type: ignore
    valid = DataLoader(valid_set, batch_size=batch_size, shuffle=True) # type: ignore
    test = DataLoader(test_set, batch_size=batch_size, shuffle=True) # type: ignore

    return train, valid, test


# %% def single label model
class Model_s(nn.Module):
    def __init__(self, hidden_size, num_layers, activate):
        super(Model_s, self).__init__()
        self.flatten = nn.Flatten()
        layers = []
        layers.append(nn.Linear(N_FEATURES, hidden_size))
        if activate == True:
             layers.append(nn.ReLU())
        for i in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size, 1))
        layers.append(nn.Sigmoid())
        self.linear_sigmoid_stack = nn.Sequential(*layers)

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_sigmoid_stack(x)
        return logits 
    

# %% def model training
def model_training_s(train, valid, model, lam, learning_rate, epochs):
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
    train_loss_list = []
    valid_loss_list = []

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")

        # train loss loop
        train_loss = 0
        for X, Y in train:
            batch_loss = nn.functional.binary_cross_entropy(model(X), Y[:, 0].unsqueeze(1))

            # L1 regularization
            regularization_loss = 0
            for param in model.parameters():
                regularization_loss += torch.sum(torch.abs(param))
            batch_loss += lam*regularization_loss

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            train_loss += batch_loss.item()

        train_loss /= len(train)
        train_loss_list.append(train_loss)
    

        if plot:
            # valid loss loop
            valid_loss = 0
            with torch.no_grad():
                for X, Y in valid:
                    loss = nn.functional.binary_cross_entropy(model(X), Y[:, 0].unsqueeze(1))
                    valid_loss += loss.item()

            valid_loss /= len(valid)
            valid_loss_list.append(valid_loss)

    if plot:
    # loss plot
        x_axix = np.arange(epochs)
        plt.figure()
        plt.plot(x_axix, train_loss_list, color="red", label="train loss")
        plt.plot(x_axix, valid_loss_list, color="blue", label="valid loss")
        plt.xlabel("Epochs")
        plt.legend()
        plt.show()
    
    return model

# %% def performance plot
def performance(labels, pred):    
    # AUROC + PR
    auc = roc_auc_score(labels, pred)
    ap = average_precision_score(labels, pred)

    if plot:

        plt.style.use('seaborn')
        sns.set_style("whitegrid")
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

        fpr, tpr, _ = roc_curve(labels, pred)
        precision, recall, _ = precision_recall_curve(labels, pred)

        ax[0].plot(fpr, tpr, label='AUC = %.3f' % auc)
        ax[0].set_xlim([-.01, 1.01]) # type: ignore
        ax[0].set_ylim([-.01, 1.01]) # type: ignore
        ax[0].set_xlabel('False Positive Rate (1 - Specificity)', fontsize=14)
        ax[0].set_ylabel('True Positive Rate (Sensitivity)', fontsize=14)
        ax[0].plot([0, 1], [0, 1], 'k--', label='No information')
        ax[0].legend(loc='lower right', fontsize=14)

        ax[1].plot(recall, precision, label='Avg Precision = %.3f' % ap)
        ax[1].set_xlim([-.01, 1.01]) # type: ignore
        ax[1].set_ylim([-.01, 1.01]) # type: ignore
        ax[1].set_xlabel('Recall (Sensitivity)', fontsize=14)
        ax[1].set_ylabel('Precision (Positive Predictive Value)', fontsize=14)
        ax[1].plot([0, 1], [labels.mean(), labels.mean()], 'k--', label='No information') # type: ignore
        ax[1].legend(loc='upper right', fontsize=14)

        plt.show()

    return auc, ap

# %% def single modeling
def single_modeling(train, valid, test):
    model = Model_s(hidden_size, num_layers, activate)

    model = model_training_s(train, valid, model, lam, learning_rate, epochs)

    labels = []
    pred = []
    with torch.no_grad():
        torch.manual_seed(2023)
        for X, Y in test:
            pred += model(X).tolist()
            labels += Y[:, 0].tolist()
    pred = sum(pred, [])

    auc, ap = performance(np.array([labels[i] for i in range(len(labels))]), np.array([pred[i] for i in range(len(pred))]))

    return auc, ap



# %% ---------------multi-label model--------------
# %% def MTLnet model
class Model_m(nn.Module):
    def __init__(self, hidden_size, num_layers, activate):
        super(Model_m, self).__init__()
        self.flatten = nn.Flatten()
        layers = []
        layers.append(nn.Linear(N_FEATURES, hidden_size))
        if activate == True:
             layers.append(nn.ReLU())
        for i in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size, 2))
        layers.append(nn.Sigmoid())
        self.linear_sigmoid_stack = nn.Sequential(*layers)

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_sigmoid_stack(x)
        return logits 


# %% def model training
def model_training_m(train, valid, model, lam, learning_rate, epochs):
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    train_loss_list = []
    valid_loss_list = []

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        
        train_loss = 0
        for X, Y in train:
            batch_loss = nn.functional.binary_cross_entropy(model(X)[:, 0], Y[:, 0])
            batch_loss += nn.functional.binary_cross_entropy(model(X)[:, 1], Y[:, 1])
            batch_loss /= 2

            # L1 regularization
            regularization_loss = 0
            for param in model.parameters():
                regularization_loss += torch.sum(torch.abs(param))
            batch_loss += lam*regularization_loss

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            train_loss += batch_loss.item()

        if plot:
            train_loss /= len(train)
            train_loss_list.append(train_loss)
            
            valid_loss = 0
            with torch.no_grad():
                for X, Y in valid:
                    loss = nn.functional.binary_cross_entropy(model(X)[:, 0], Y[:, 0])
                    loss += nn.functional.binary_cross_entropy(model(X)[:, 1], Y[:, 1])
                    loss /= 2
                    valid_loss += loss

            valid_loss /= len(valid)
            valid_loss_list.append(valid_loss)

    if plot:
        # loss plot
        x_axix = np.arange(epochs)
        plt.figure()
        plt.plot(x_axix, train_loss_list, color="red", label="train loss")
        plt.plot(x_axix, valid_loss_list, color="blue", label="valid loss")
        plt.xlabel("Epochs")
        plt.legend()
        plt.show()

    return model

# %% def multi modeling
def multi_modeling(train, valid, test):
    print(f"hidden_size: {hidden_size}")

    model = Model_m(hidden_size, num_layers, activate)

    model = model_training_m(train, valid, model, lam, learning_rate, epochs)

    label = []
    pred = []
    with torch.no_grad():
        torch.manual_seed(2023)
        for X, Y in test:
            pred += model(X).tolist()
            label += Y.tolist()

    auc, ap = performance(np.array([label[i][0] for i in range(len(label))]),
                np.array([pred[j][0] for j in range(len(pred))]))
    
    return auc, ap

# %% simulate
# %%  ---------------one-node test--------------
results = pd.DataFrame(columns=["ReLu", "method", "AUC", "AP"])
for i in range(10):
          x, e1, e2 = generate_data(event_rate, similarity)
          train, valid, test = load_data(x, e1, e2)
          for activate in [True, False]:
            auc_single, ap_single = single_modeling(train, valid, test)
            auc_multi, ap_multi = multi_modeling(train, valid, test)
            
            results.loc[len(results)] = {"ReLu": activate,
                                         "method": "single",
                                         "AUC": auc_single,
                                         "AP": ap_single}
            
            results.loc[len(results)] = {"ReLu": activate,
                                         "method": "multi",
                                         "AUC": auc_multi,
                                         "AP": ap_multi}
            print(activate, results)

            # results.to_csv("results_running_10000.csv")     

# results.to_csv("results_final_10000.csv")


# %%  ---------------comparision plot--------------
def compare(results):
    results["method-ReLu"] = results["method"] +""+ results["ReLu"].astype(str)
    sns.set(style="ticks")
    # sns.relplot(x='method-ReLu', y='AUC', hue='method', data=result, alpha=0.5)
    sns.boxplot(x='method-ReLu', y='AUC', hue='method', data=results)
    plt.show()
    # sns.relplot(x='method-ReLu', y='AP', hue='method', data=result, alpha=0.5)
    sns.boxplot(x='method-ReLu', y='AP', hue='method', data=results)
    plt.show()
# %%
