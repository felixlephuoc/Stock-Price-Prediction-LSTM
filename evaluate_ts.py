
# Script to evaluate the functions
import numpy as np
from matplotlib import pylab as plt
from tools import matrix_to_array

def evaluate_ts(features, y_true, y_pred):
    print("Evaluation of the predictions:")
    print("MSE (mean squared error): ", np.mean(np.square(y_true - y_pred)))
    print("MAE (mean absolute error): ", np.mean(np.abs(y_true - y_pred)))

    print("Benchmark: if prediction == last feature (prediction without any model)")
    print("MSE: ", np.mean(np.square(features[:,-1] - y_true)))
    print("MAE: ", np.mean(np.abs(features[:,-1] - y_true)))

    plt.plot(matrix_to_array(y_true), 'b')
    plt.plot(matrix_to_array(y_pred), 'r')
    plt.xlabel("Days")
    plt.ylabel("Stock Price (USD)")
    plt.title("Predicted (Red) VS Real (Blue) on test set - LSTM")
    plt.savefig("./graph/rnn/predicted_vs_real_values_test_set.png")
    plt.show()

    error = np.abs(matrix_to_array(y_pred) - matrix_to_array(y_true))
    plt.plot(error, 'r')
    fit = np.polyfit(range(len(error)), error, deg=1)
    plt.plot(fit[0] * range(len(error))+ fit[1], '--')
    plt.xlabel("Days")
    plt.ylabel("Prediction error L1 norm")
    plt.title("Prediction error (absolute) and trendline - LSTM")
    plt.savefig("./graph/rnn/predictions_error_trendline.png")
    plt.show()