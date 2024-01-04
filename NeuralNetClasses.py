from torch import nn


class Multiclass1(nn.Module):
    def __init__(self, NUM_FEATURES, NUM_CLASSES, activation_fcn, a):
        super().__init__()
        self.hidden1 = nn.Linear(NUM_FEATURES, a)
        if activation_fcn == "relu":
            self.act1 = nn.ReLU()
        else:
            self.act1 = nn.Tanh()
        self.output = nn.Linear(a, NUM_CLASSES)

    def forward(self, x):
        x = self.act1(self.hidden1(x))
        x = self.output(x)
        return x


class Multiclass2(nn.Module):
    def __init__(self, NUM_FEATURES, NUM_CLASSES, activation_fcn, a, b):
        super().__init__()
        self.hidden1 = nn.Linear(NUM_FEATURES, a)
        if activation_fcn == "relu":
            self.act1 = nn.ReLU()
        else:
            self.act1 = nn.Tanh()
        self.hidden2 = nn.Linear(a, b)
        if activation_fcn == "relu":
            self.act2 = nn.ReLU()
        else:
            self.act2 = nn.Tanh()
        self.output = nn.Linear(b, NUM_CLASSES)

    def forward(self, x):
        x = self.act1(self.hidden1(x))
        x = self.act2(self.hidden2(x))
        x = self.output(x)
        return x


class Multiclass3(nn.Module):
    def __init__(self, NUM_FEATURES, NUM_CLASSES, activation_fcn, a, b, c):
        super().__init__()
        self.hidden1 = nn.Linear(NUM_FEATURES, a)
        if activation_fcn == "relu":
            self.act1 = nn.ReLU()
        else:
            self.act1 = nn.Tanh()
        self.hidden2 = nn.Linear(a, b)
        if activation_fcn == "relu":
            self.act2 = nn.ReLU()
        else:
            self.act2 = nn.Tanh()

        self.hidden3 = nn.Linear(b, c)
        if activation_fcn == "relu":
            self.act3 = nn.ReLU()
        else:
            self.act3 = nn.Tanh()

        self.output = nn.Linear(c, NUM_CLASSES)

    def forward(self, x):
        x = self.act1(self.hidden1(x))
        x = self.act2(self.hidden2(x))
        x = self.act3(self.hidden3(x))
        x = self.output(x)
        return x


class Multiclass4(nn.Module):
    def __init__(self, NUM_FEATURES, NUM_CLASSES, activation_fcn, a, b, c, d):
        super().__init__()
        self.hidden1 = nn.Linear(NUM_FEATURES, a)
        if activation_fcn == "relu":
            self.act1 = nn.ReLU()
        else:
            self.act1 = nn.Tanh()
        self.hidden2 = nn.Linear(a, b)
        if activation_fcn == "relu":
            self.act2 = nn.ReLU()
        else:
            self.act2 = nn.Tanh()

        self.hidden3 = nn.Linear(b, c)
        if activation_fcn == "relu":
            self.act3 = nn.ReLU()
        else:
            self.act3 = nn.Tanh()

        self.hidden4 = nn.Linear(c, d)
        if activation_fcn == "relu":
            self.act4 = nn.ReLU()
        else:
            self.act4 = nn.Tanh()

        self.output = nn.Linear(d, NUM_CLASSES)

    def forward(self, x):
        x = self.act1(self.hidden1(x))
        x = self.act2(self.hidden2(x))
        x = self.act3(self.hidden3(x))
        x = self.act4(self.hidden4(x))
        x = self.output(x)
        return x


class Multiclass5(nn.Module):
    def __init__(self, NUM_FEATURES, NUM_CLASSES, activation_fcn, a, b, c, d, e):
        super().__init__()
        self.hidden1 = nn.Linear(NUM_FEATURES, a)
        if activation_fcn == "relu":
            self.act1 = nn.ReLU()
        else:
            self.act1 = nn.Tanh()
        self.hidden2 = nn.Linear(a, b)
        if activation_fcn == "relu":
            self.act2 = nn.ReLU()
        else:
            self.act2 = nn.Tanh()

        self.hidden3 = nn.Linear(b, c)
        if activation_fcn == "relu":
            self.act3 = nn.ReLU()
        else:
            self.act3 = nn.Tanh()

        self.hidden4 = nn.Linear(c, d)
        if activation_fcn == "relu":
            self.act4 = nn.ReLU()
        else:
            self.act4 = nn.Tanh()

        self.hidden5 = nn.Linear(d, e)
        if activation_fcn == "relu":
            self.act5 = nn.ReLU()
        else:
            self.act5 = nn.Tanh()

        self.output = nn.Linear(e, NUM_CLASSES)

    def forward(self, x):
        x = self.act1(self.hidden1(x))
        x = self.act2(self.hidden2(x))
        x = self.act3(self.hidden3(x))
        x = self.act4(self.hidden4(x))
        x = self.act5(self.hidden5(x))
        x = self.output(x)
        return x


class Multiclass6(nn.Module):
    def __init__(self, NUM_FEATURES, NUM_CLASSES, activation_fcn, a, b, c, d, e, f):
        super().__init__()
        self.hidden1 = nn.Linear(NUM_FEATURES, a)
        if activation_fcn == "relu":
            self.act1 = nn.ReLU()
        else:
            self.act1 = nn.Tanh()
        self.hidden2 = nn.Linear(a, b)
        if activation_fcn == "relu":
            self.act2 = nn.ReLU()
        else:
            self.act2 = nn.Tanh()

        self.hidden3 = nn.Linear(b, c)
        if activation_fcn == "relu":
            self.act3 = nn.ReLU()
        else:
            self.act3 = nn.Tanh()

        self.hidden4 = nn.Linear(c, d)
        if activation_fcn == "relu":
            self.act4 = nn.ReLU()
        else:
            self.act4 = nn.Tanh()

        self.hidden5 = nn.Linear(d, e)
        if activation_fcn == "relu":
            self.act5 = nn.ReLU()
        else:
            self.act5 = nn.Tanh()

        self.hidden6 = nn.Linear(e, f)
        if activation_fcn == "relu":
            self.act6 = nn.ReLU()
        else:
            self.act6 = nn.Tanh()

        self.output = nn.Linear(f, NUM_CLASSES)

    def forward(self, x):
        x = self.act1(self.hidden1(x))
        x = self.act2(self.hidden2(x))
        x = self.act3(self.hidden3(x))
        x = self.act4(self.hidden4(x))
        x = self.act5(self.hidden5(x))
        x = self.act6(self.hidden6(x))
        x = self.output(x)
        return x


class Multiclass7(nn.Module):
    def __init__(self, NUM_FEATURES, NUM_CLASSES, activation_fcn, a, b, c, d, e, f, g):
        super().__init__()
        self.hidden1 = nn.Linear(NUM_FEATURES, a)
        if activation_fcn == "relu":
            self.act1 = nn.ReLU()
        else:
            self.act1 = nn.Tanh()
        self.hidden2 = nn.Linear(a, b)
        if activation_fcn == "relu":
            self.act2 = nn.ReLU()
        else:
            self.act2 = nn.Tanh()

        self.hidden3 = nn.Linear(b, c)
        if activation_fcn == "relu":
            self.act3 = nn.ReLU()
        else:
            self.act3 = nn.Tanh()

        self.hidden4 = nn.Linear(c, d)
        if activation_fcn == "relu":
            self.act4 = nn.ReLU()
        else:
            self.act4 = nn.Tanh()

        self.hidden5 = nn.Linear(d, e)
        if activation_fcn == "relu":
            self.act5 = nn.ReLU()
        else:
            self.act5 = nn.Tanh()

        self.hidden6 = nn.Linear(e, f)
        if activation_fcn == "relu":
            self.act6 = nn.ReLU()
        else:
            self.act6 = nn.Tanh()

        self.hidden7 = nn.Linear(f, g)
        if activation_fcn == "relu":
            self.act7 = nn.ReLU()
        else:
            self.act7 = nn.Tanh()

        self.output = nn.Linear(g, NUM_CLASSES)

    def forward(self, x):
        x = self.act1(self.hidden1(x))
        x = self.act2(self.hidden2(x))
        x = self.act3(self.hidden3(x))
        x = self.act4(self.hidden4(x))
        x = self.act5(self.hidden5(x))
        x = self.act6(self.hidden6(x))
        x = self.act7(self.hidden7(x))
        x = self.output(x)
        return x


class Multiclass8(nn.Module):
    def __init__(self, NUM_FEATURES, NUM_CLASSES, activation_fcn, a, b, c, d, e, f, g, h):
        super().__init__()
        self.hidden1 = nn.Linear(NUM_FEATURES, a)
        if activation_fcn == "relu":
            self.act1 = nn.ReLU()
        else:
            self.act1 = nn.Tanh()
        self.hidden2 = nn.Linear(a, b)
        if activation_fcn == "relu":
            self.act2 = nn.ReLU()
        else:
            self.act2 = nn.Tanh()

        self.hidden3 = nn.Linear(b, c)
        if activation_fcn == "relu":
            self.act3 = nn.ReLU()
        else:
            self.act3 = nn.Tanh()

        self.hidden4 = nn.Linear(c, d)
        if activation_fcn == "relu":
            self.act4 = nn.ReLU()
        else:
            self.act4 = nn.Tanh()

        self.hidden5 = nn.Linear(d, e)
        if activation_fcn == "relu":
            self.act5 = nn.ReLU()
        else:
            self.act5 = nn.Tanh()

        self.hidden6 = nn.Linear(e, f)
        if activation_fcn == "relu":
            self.act6 = nn.ReLU()
        else:
            self.act6 = nn.Tanh()

        self.hidden7 = nn.Linear(f, g)
        if activation_fcn == "relu":
            self.act7 = nn.ReLU()
        else:
            self.act7 = nn.Tanh()

        self.hidden8 = nn.Linear(g, h)
        if activation_fcn == "relu":
            self.act8 = nn.ReLU()
        else:
            self.act8 = nn.Tanh()

        self.output = nn.Linear(h, NUM_CLASSES)

    def forward(self, x):
        x = self.act1(self.hidden1(x))
        x = self.act2(self.hidden2(x))
        x = self.act3(self.hidden3(x))
        x = self.act4(self.hidden4(x))
        x = self.act5(self.hidden5(x))
        x = self.act6(self.hidden6(x))
        x = self.act7(self.hidden7(x))
        x = self.act8(self.hidden8(x))
        x = self.output(x)
        return x


class Multiclass9(nn.Module):
    def __init__(self, NUM_FEATURES, NUM_CLASSES, activation_fcn, a, b, c, d, e, f, g, h, i):
        super().__init__()
        self.hidden1 = nn.Linear(NUM_FEATURES, a)
        if activation_fcn == "relu":
            self.act1 = nn.ReLU()
        else:
            self.act1 = nn.Tanh()
        self.hidden2 = nn.Linear(a, b)
        if activation_fcn == "relu":
            self.act2 = nn.ReLU()
        else:
            self.act2 = nn.Tanh()

        self.hidden3 = nn.Linear(b, c)
        if activation_fcn == "relu":
            self.act3 = nn.ReLU()
        else:
            self.act3 = nn.Tanh()

        self.hidden4 = nn.Linear(c, d)
        if activation_fcn == "relu":
            self.act4 = nn.ReLU()
        else:
            self.act4 = nn.Tanh()

        self.hidden5 = nn.Linear(d, e)
        if activation_fcn == "relu":
            self.act5 = nn.ReLU()
        else:
            self.act5 = nn.Tanh()

        self.hidden6 = nn.Linear(e, f)
        if activation_fcn == "relu":
            self.act6 = nn.ReLU()
        else:
            self.act6 = nn.Tanh()

        self.hidden7 = nn.Linear(f, g)
        if activation_fcn == "relu":
            self.act7 = nn.ReLU()
        else:
            self.act7 = nn.Tanh()

        self.hidden8 = nn.Linear(g, h)
        if activation_fcn == "relu":
            self.act8 = nn.ReLU()
        else:
            self.act8 = nn.Tanh()

        self.hidden9 = nn.Linear(h, i)
        if activation_fcn == "relu":
            self.act9 = nn.ReLU()
        else:
            self.act9 = nn.Tanh()

        self.output = nn.Linear(i, NUM_CLASSES)

    def forward(self, x):
        x = self.act1(self.hidden1(x))
        x = self.act2(self.hidden2(x))
        x = self.act3(self.hidden3(x))
        x = self.act4(self.hidden4(x))
        x = self.act5(self.hidden5(x))
        x = self.act6(self.hidden6(x))
        x = self.act7(self.hidden7(x))
        x = self.act8(self.hidden8(x))
        x = self.act9(self.hidden9(x))
        x = self.output(x)
        return x


class Multiclass10(nn.Module):
    def __init__(self, NUM_FEATURES, NUM_CLASSES, activation_fcn, a, b, c, d, e, f, g, h, i, j):
        super().__init__()
        self.hidden1 = nn.Linear(NUM_FEATURES, a)
        if activation_fcn == "relu":
            self.act1 = nn.ReLU()
        else:
            self.act1 = nn.Tanh()
        self.hidden2 = nn.Linear(a, b)
        if activation_fcn == "relu":
            self.act2 = nn.ReLU()
        else:
            self.act2 = nn.Tanh()

        self.hidden3 = nn.Linear(b, c)
        if activation_fcn == "relu":
            self.act3 = nn.ReLU()
        else:
            self.act3 = nn.Tanh()

        self.hidden4 = nn.Linear(c, d)
        if activation_fcn == "relu":
            self.act4 = nn.ReLU()
        else:
            self.act4 = nn.Tanh()

        self.hidden5 = nn.Linear(d, e)
        if activation_fcn == "relu":
            self.act5 = nn.ReLU()
        else:
            self.act5 = nn.Tanh()

        self.hidden6 = nn.Linear(e, f)
        if activation_fcn == "relu":
            self.act6 = nn.ReLU()
        else:
            self.act6 = nn.Tanh()

        self.hidden7 = nn.Linear(f, g)
        if activation_fcn == "relu":
            self.act7 = nn.ReLU()
        else:
            self.act7 = nn.Tanh()

        self.hidden8 = nn.Linear(g, h)
        if activation_fcn == "relu":
            self.act8 = nn.ReLU()
        else:
            self.act8 = nn.Tanh()

        self.hidden9 = nn.Linear(h, i)
        if activation_fcn == "relu":
            self.act9 = nn.ReLU()
        else:
            self.act9 = nn.Tanh()

        self.hidden10 = nn.Linear(i, j)
        if activation_fcn == "relu":
            self.act10 = nn.ReLU()
        else:
            self.act10 = nn.Tanh()

        self.output = nn.Linear(j, NUM_CLASSES)

    def forward(self, x):
        x = self.act1(self.hidden1(x))
        x = self.act2(self.hidden2(x))
        x = self.act3(self.hidden3(x))
        x = self.act4(self.hidden4(x))
        x = self.act5(self.hidden5(x))
        x = self.act6(self.hidden6(x))
        x = self.act7(self.hidden7(x))
        x = self.act8(self.hidden8(x))
        x = self.act9(self.hidden9(x))
        x = self.act10(self.hidden10(x))
        x = self.output(x)
        return x
