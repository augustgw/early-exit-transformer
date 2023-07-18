import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("model_1_dir", type=str, help="Name of first model dir in trained_model/")
parser.add_argument("model_2_dir", type=str, help="Name of second model dir in trained_model/")
parser.add_argument("plot_name", type=str, help="Name of for plot file")
args = parser.parse_args()

lin_arr = np.genfromtxt('trained_model/' + args.model_1_dir + '/train_loss.txt', delimiter=',')
lstm_arr = np.genfromtxt('trained_model/' + args.model_2_dir + '/train_loss.txt', delimiter=',')

# Append LSTM results and convert type to float
lstm_arr = lstm_arr[:, 1]
lstm_arr = np.reshape(lstm_arr, (len(lstm_arr), 1))
lin_arr = np.append(lin_arr, lstm_arr, axis=1)
lin_arr = lin_arr[:, 1:]
lin_arr = lin_arr.astype(float)

# Normalize
min_val = np.min(lin_arr)
max_val = np.max(lin_arr)
lin_arr_normalized = (lin_arr - min_val) / (max_val - min_val)

fig, ax = plt.subplots()
ax.plot(lin_arr_normalized[:,0], label=f'Linear')
ax.plot(lin_arr_normalized[:,1], label=f'LSTM')

ax.set_xlabel('Epoch')
ax.set_ylabel('Loss (normalized)')
ax.legend()
plt.savefig('trained_model/' + args.plot_name)
