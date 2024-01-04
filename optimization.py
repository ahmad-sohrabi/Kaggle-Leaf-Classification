import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.model_selection import KFold

from CustomFunctions import *


def train_custom_model_2(params, show_plot):
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

    kfold_losses = []
    kfold = KFold(n_splits=10, shuffle=True)
    for fold, (train_ids, test_ids) in enumerate(kfold.split(X)):
        X_train, X_test, y_train, y_test = X[train_ids, :], X[test_ids, :], y[train_ids], y[test_ids]

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
            y_pred = torch.softmax(y_logits, dim=1).argmax(
                dim=1)
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

        kfold_losses.append(test_loss.cpu())
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
    return np.mean(kfold_losses)


def objective_function(trial):
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-1)
    activation_fcn = trial.suggest_categorical('activation_fcn', choices=["relu", "tanh"])
    norm_method = trial.suggest_categorical('norm_method', choices=["standard", "normal"])
    """model_type = trial.suggest_categorical('model_type',
                                           choices=["model2", "model3", "model4", "model5", "model6", "model7",
                                                    "model8", "model9", "model10"])"""

    model_type = trial.suggest_categorical('model_type',
                                           choices=["model2", "model3",
                                                    "model4"])
    a = trial.suggest_int("a", 10, 512)
    b = trial.suggest_int("b", 10, 512)
    c = trial.suggest_int("c", 10, 512)
    d = trial.suggest_int("d", 10, 512)
    e = trial.suggest_int("e", 10, 512)
    f = trial.suggest_int("f", 10, 512)
    g = trial.suggest_int("g", 10, 512)
    h = trial.suggest_int("h", 10, 512)
    i = trial.suggest_int("i", 10, 512)
    j = trial.suggest_int("j", 10, 512)

    params = {'learning_rate': learning_rate,
              'activation_fcn': activation_fcn,
              'norm_method': norm_method,
              'model_type': model_type,
              'a': a,
              'b': b,
              'c': c,
              'd': d,
              'e': e,
              'f': f,
              'g': g,
              'h': h,
              'i': i,
              'j': j,
              }

    test_loss = train_custom_model_2(params, False)
    return test_loss


device = "cuda" if torch.cuda.is_available() else "cpu"

study = optuna.create_study(pruner=MedianPruner(), sampler=TPESampler(), direction="minimize")
study.optimize(objective_function, n_trials=2000, n_jobs=1)
best_params = study.best_params
print(f"{best_params}")

test_loss, model, test_acc = train_custom_model(best_params, True)
print(f"Test Loss is {test_loss}")
print(f"Test Accuracy is {test_acc}")


# Obtaining Classes of the problem
data = pd.read_csv("train.csv")
data.fillna(method='ffill', inplace=True)
y = data.iloc[:, 1].values
y = y.reshape(-1, 1)
y_shape = y.shape
encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)

classes = list(encoder.classes_)

# Loading Kaggle Test Data
test_data = pd.read_csv("test.csv")

X_test = test_data.iloc[:, 1:].values

test_ids = test_data.pop('id')

if best_params["norm_method"] == "standard":
    scaler = StandardScaler()
else:
    scaler = MinMaxScaler()

scaler.fit(X_test)
X_test = scaler.transform(X_test)

X_test = torch.from_numpy(X_test).type(torch.float)

X_test = X_test.to(device)

model.eval()
with torch.inference_mode():
    test_logits = model(X_test)
    test_pred = torch.softmax(test_logits, dim=1)
    test_pred = test_pred.tolist()
    test_pred = np.array(test_pred)
    submission = pd.DataFrame(test_pred, index=test_ids, columns=encoder.classes_)
    submission.to_csv('submission.csv')
