from fcnn import fcnn_closure
from lstm import lstm_closure
from cnn import cnn_closure
from model_profiler import model_profiler
model_builders = [
    fcnn_closure(n_hidden=1, n_neurons=64, learning_rate=1e-4, activation="relu"),
    cnn_closure(learning_rate=1e-4),
    lstm_closure()
]

for model in model_builders:
    profile = model_profiler(model, 16)
    print(profile)
