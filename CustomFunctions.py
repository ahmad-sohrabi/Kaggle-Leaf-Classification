from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
from NeuralNetClasses import *
import torch.optim as optim
import torch
import pandas as pd
import numpy as np


def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc


def train_custom_model(params, show_plot):
    data = pd.read_csv("train.csv")
    data.fillna(method='ffill', inplace=True)
    X = data.iloc[:, 2:].values
    y = data.iloc[:, 1].values

    learning_rate = params['learning_rate']
    activation_fcn = params['activation_fcn']
    norm_method = params['norm_method']
    model_type = params['model_type']
    a = params["a"]
    b = params["b"]
    c = params["c"]
    d = params["d"]
    e = params["e"]
    f = params["f"]
    g = params["g"]
    h = params["h"]
    i = params["i"]
    j = params["j"]

    if norm_method == "standard":
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()

    scaler.fit(X)
    X = scaler.transform(X)

    y = y.reshape(-1, 1)
    encoder = LabelEncoder()
    encoder.fit(y)
    y = encoder.transform(y)

    NUM_CLASSES = len(encoder.classes_)
    NUM_FEATURES = X.shape[1]
    RANDOM_SEED = 42

    X = torch.from_numpy(X).type(torch.float)
    y = torch.from_numpy(y).type(torch.LongTensor)

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=RANDOM_SEED
                                                        )
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if model_type == "model1":
        model = Multiclass1(NUM_FEATURES, NUM_CLASSES, activation_fcn, a)
        model.to(device)
    elif model_type == "model2":
        model = Multiclass2(NUM_FEATURES, NUM_CLASSES, activation_fcn, a, b)
        model.to(device)
    elif model_type == "model3":
        model = Multiclass3(NUM_FEATURES, NUM_CLASSES, activation_fcn, a, b, c)
        model.to(device)
    elif model_type == "model4":
        model = Multiclass4(NUM_FEATURES, NUM_CLASSES, activation_fcn, a, b, c, d)
        model.to(device)
    elif model_type == "model5":
        model = Multiclass5(NUM_FEATURES, NUM_CLASSES, activation_fcn, a, b, c, d, e)
        model.to(device)
    elif model_type == "model6":
        model = Multiclass6(NUM_FEATURES, NUM_CLASSES, activation_fcn, a, b, c, d, e, f)
        model.to(device)
    elif model_type == "model7":
        model = Multiclass7(NUM_FEATURES, NUM_CLASSES, activation_fcn, a, b, c, d, e, f, g)
        model.to(device)
    elif model_type == "model8":
        model = Multiclass8(NUM_FEATURES, NUM_CLASSES, activation_fcn, a, b, c, d, e, f, g, h)
        model.to(device)
    elif model_type == "model9":
        model = Multiclass9(NUM_FEATURES, NUM_CLASSES, activation_fcn, a, b, c, d, e, f, g, h, i)
        model.to(device)
    elif model_type == "model10":
        model = Multiclass10(NUM_FEATURES, NUM_CLASSES, activation_fcn, a, b, c, d, e, f, g, h, i, j)
        model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)

    epochs = 1000

    X_train, y_train = X_train.to(device), y_train.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)
    test_loss_array = np.array([])
    train_loss_array = np.array([])

    test_accuracy_array = np.array([])
    train_accuracy_array = np.array([])
    for epoch in range(epochs):
        model.train()

        y_logits = model(X_train)  # model outputs raw logits
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)
        loss = loss_fn(y_logits, y_train)

        acc = accuracy_fn(y_true=y_train,
                          y_pred=y_pred)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        model.eval()
        with torch.inference_mode():
            test_logits = model(X_test)
            test_pred = torch.softmax(test_logits, dim=1).argmax(dim=1)
            test_loss = loss_fn(test_logits, y_test)
            test_acc = accuracy_fn(y_true=y_test,
                                   y_pred=test_pred)
            test_loss_array = np.append(test_loss_array, test_loss.cpu())
            train_loss_array = np.append(train_loss_array, loss.cpu())

            test_accuracy_array = np.append(test_accuracy_array, test_acc)
            train_accuracy_array = np.append(train_accuracy_array, acc)

    if show_plot:
        plt.plot(range(len(train_loss_array)), train_loss_array, label='train loss')
        plt.plot(range(len(test_loss_array)), test_loss_array, label='test loss')
        plt.ylabel("Loss")
        plt.xlabel("Epochs")
        plt.legend()
        plt.show()

        plt.plot(range(len(train_accuracy_array)), train_accuracy_array, label='train accuracy')
        plt.plot(range(len(test_accuracy_array)), test_accuracy_array, label='test accuracy')
        plt.ylabel("Accuracy")
        plt.xlabel("Epochs")
        plt.legend()
        plt.show()
    return test_loss, model, test_acc


class LeafDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        if not torch.is_tensor(X) and not torch.is_tensor(y):
            self.X = torch.from_numpy(X).type(torch.float)
            self.y = torch.from_numpy(y).type(torch.LongTensor)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]
