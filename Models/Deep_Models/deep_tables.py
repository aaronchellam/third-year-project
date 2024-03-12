from Util.augmentation import load_augmented_data, load_deep_augmented_C
from Util.util import dict_to_df, get_analysis_df
from Util.deep_logger import load_deep_data
from Util.deep_skfold import deep_cv_score
from fcnn import fcnn_closure
from lstm import lstm_closure
from cnn import cnn_closure
import pandas as pd


# TODO: Selu activation with "lecun_normal"

def write_latex(file, dfs):
    for df in dfs:
        f.write(df.to_latex(bold_rows=True))
        f.write("\n")
        f.write(r"\hfill \break")
        f.write(r"\hfill \break")
        f.write("\n")


f = open("deep_tables.tex", 'w')

#TODO Select correct data.
#TODO: Restore batch size
# d1X, d1y, d2X, d2y, d3X, d3y = load_deep_data()
d1X, d1y, d2X, d2y, d3X, d3y = load_augmented_data(deep=True)
augCX, augCy = load_deep_augmented_C()
# datasets = [[d1X, d1y], [d2X, d2y], [d3X, d3y]]
datasets = [[augCX, augCy]]
classifiers = {0: "FCNN", 1: "CNN", 2: "LSTM"}

model_builders = [
    fcnn_closure(n_hidden=1, n_neurons=64, learning_rate=1e-4, activation="relu"),
    cnn_closure(learning_rate=1e-4)
    # cnn_closure(learning_rate=1e-4, n_hidden=1, dropout=True),
    # lstm_closure()
    # fcnn_closure(n_hidden=1, n_neurons=64, learning_rate=1e-4, activation="selu", kernel_initializer="lecun_normal")
]

names = ["FCNN", "CNN"]
# names = [ "FCNN (SELU)"]

for i1, dataset in enumerate(datasets):
    X, y = dataset
    f.write(f"Results for Dataset {i1 + 1} \n")
    f.write(r"\\ \hfill \break")

    results_dict = {}
    analysis_dict = {}
    total_accuracies = []
    training_times = []
    param_counts = []

    # TODO: Remove - selects only D3
    # if i1 != 2:
    #     continue


    for i2, builder in enumerate(model_builders):

        # TODO: Affects model selection.
        # if i2 == 2:
        #     continue
        print(f"\nTraining Model: {names[i2]} on dataset {i1+1}")

        # Main training line.
        accuracies, total_accuracy, total_training_time, total_params = deep_cv_score(X, y, builder, batch_size=16,
                                                                                      epochs=50, n_splits=5)
        results_dict[classifiers[i2]] = accuracies
        analysis_dict[classifiers[i2]] = [total_training_time, total_params]
        total_accuracies.append(total_accuracy)
    # training_times.append(total_training_time)
    # param_counts.append(total_params)

    df_results = dict_to_df(results_dict, total_accuracies)
    df_analysis = get_analysis_df(analysis_dict)

    write_latex(f, [df_results, df_analysis])

f.close()
