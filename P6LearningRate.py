from CustomFunctions import *

params = {'learning_rate': 0.0001, 'activation_fcn': 'tanh', 'norm_method': 'standard', 'model_type': 'model2', 'a': 32, 'b': 32, 'c': 0, 'd': 0, 'e': 0, 'f': 0, 'g': 0, 'h': 0, 'i': 0, 'j': 0}
test_loss, model, acc = train_custom_model(params, True)
print(f"Test Loss is {test_loss}")
print(f"Test Accuracy is {acc}")

params = {'learning_rate': 0.001, 'activation_fcn': 'tanh', 'norm_method': 'standard', 'model_type': 'model2', 'a': 32, 'b': 32, 'c': 0, 'd': 0, 'e': 0, 'f': 0, 'g': 0, 'h': 0, 'i': 0, 'j': 0}
test_loss, model, acc = train_custom_model(params, True)
print(f"Test Loss is {test_loss}")
print(f"Test Accuracy is {acc}")

params = {'learning_rate': 0.01, 'activation_fcn': 'tanh', 'norm_method': 'standard', 'model_type': 'model2', 'a': 32, 'b': 32, 'c': 0, 'd': 0, 'e': 0, 'f': 0, 'g': 0, 'h': 0, 'i': 0, 'j': 0}
test_loss, model, acc = train_custom_model(params, True)
print(f"Test Loss is {test_loss}")
print(f"Test Accuracy is {acc}")