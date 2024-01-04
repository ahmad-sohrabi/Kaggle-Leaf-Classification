from CustomFunctions import *

params = {'learning_rate': 0.001, 'activation_fcn': 'tanh', 'norm_method': 'standard', 'model_type': 'model2', 'a': 30, 'b': 30, 'c': 0, 'd': 0, 'e': 0, 'f': 0, 'g': 0, 'h': 0, 'i': 0, 'j': 0}
test_loss, model, acc = train_custom_model(params, True)
print(f"Test Loss is {test_loss}")
print(f"Test Accuracy is {acc}")

params = {'learning_rate': 0.001, 'activation_fcn': 'tanh', 'norm_method': 'standard', 'model_type': 'model4', 'a': 15, 'b': 15, 'c': 15, 'd': 15, 'e': 0, 'f': 0, 'g': 0, 'h': 0, 'i': 0, 'j': 0}
test_loss, model, acc = train_custom_model(params, True)
print(f"Test Loss is {test_loss}")
print(f"Test Accuracy is {acc}")

params = {'learning_rate': 0.001, 'activation_fcn': 'tanh', 'norm_method': 'standard', 'model_type': 'model6', 'a': 10, 'b': 10, 'c': 10, 'd': 10, 'e': 10, 'f': 10, 'g': 0, 'h': 0, 'i': 0, 'j': 0}
test_loss, model, acc = train_custom_model(params, True)
print(f"Test Loss is {test_loss}")
print(f"Test Accuracy is {acc}")