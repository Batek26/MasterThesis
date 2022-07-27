from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras import Input
from keras import regularizers
from tensorflow import keras
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def MultiModel(n_inputs, n_outputs):
  inp = Input((n_inputs))
  x = Dense(64, activation = 'relu')(inp)
  x = Dense(64, activation = 'relu', kernel_regularizer=regularizers.l2(0.001))(x)
  x = Dense(16, activation = 'relu', kernel_regularizer=regularizers.l2(0.001))(x)
  out = Dense(n_outputs)(x)
  model = keras.Model(inputs = inp,outputs = out)
  model.compile(optimizer='adam', loss='mse', metrics=['mae'])
  return model

def MultiModel1(n_inputs, n_outputs):
  inp = Input((n_inputs))
  x = Dense(128, activation = 'relu')(inp)
  x = Dense(128, activation = 'relu', kernel_regularizer=regularizers.l2(0.001))(x)
  x = Dense(64, activation = 'relu', kernel_regularizer=regularizers.l2(0.001))(x)
  x = Dense(16, activation = 'relu')(x)
  out = Dense(n_outputs)(x)
  model = keras.Model(inputs = inp,outputs = out)
  model.compile(optimizer='adam', loss='mse', metrics=['mae'])
  return model

"""Przygotowanie danych"""

raw_dataset = pd.read_csv('pomiary13112020.csv')
raw_dataset.info()
raw_dataset

dataset = raw_dataset.copy()
print(len(dataset))
dataset_orig = dataset

index_to_drop = dataset_orig[dataset_orig.Hu <= 25].index
dataset_orig = dataset_orig.drop(index=index_to_drop)

index_to_drop = dataset_orig[dataset_orig.MP1 <= 1.2].index
dataset_orig = dataset_orig.drop(index=index_to_drop)
index_to_drop = dataset_orig[dataset_orig.MP1 >= 2.4].index
dataset_orig = dataset_orig.drop(index=index_to_drop)
dataset_orig.shape
px.histogram(dataset_orig, x='MP1')

index_to_drop = dataset_orig[dataset_orig.MP2 <= 1.2].index
dataset_orig = dataset_orig.drop(index=index_to_drop)
index_to_drop = dataset_orig[dataset_orig.MP2 >= 2.4].index
dataset_orig = dataset_orig.drop(index=index_to_drop)
dataset_orig.shape
px.histogram(dataset_orig, x='MP2')

index_to_drop = dataset_orig[dataset_orig.MP3 <= 1.2].index
dataset_orig = dataset_orig.drop(index=index_to_drop)
index_to_drop = dataset_orig[dataset_orig.MP3 >= 2.4].index
dataset_orig = dataset_orig.drop(index=index_to_drop)
dataset_orig.shape
px.histogram(dataset_orig, x='MP3')

index_to_drop = dataset_orig[dataset_orig.MP4 <= 1.2].index
dataset_orig = dataset_orig.drop(index=index_to_drop)
index_to_drop = dataset_orig[dataset_orig.MP4 >= 2.4].index
dataset_orig = dataset_orig.drop(index=index_to_drop)
dataset_orig.shape
px.histogram(dataset_orig, x='MP4')

dataset_orig.shape

train_dataset = dataset_orig.sample(frac=0.8,random_state=0)
test_dataset = dataset_orig.drop(train_dataset.index)
print(f'train_dataset lenght: {len(train_dataset)}')
print(f'test_dataset lenght: {len(test_dataset)}')

train_stats = dataset_orig.describe()
train_stats.pop('TeM1')
train_stats.pop('TeC1')
train_stats.pop('TeTD')
train_stats.pop('TeD')
train_stats.pop('TeTP1')
train_stats.pop('TeP1')
train_stats.pop('TeTP2')
train_stats.pop('TeP2')
train_stats.pop('TeTP3')
train_stats.pop('TeP3')
train_stats.pop('TeTP4')
train_stats.pop('TeP4')
train_stats = train_stats.transpose()
train_stats

train_labels = train_dataset.copy()
train_labels.pop('MP1')
train_labels.pop('MP2')
train_labels.pop('MP3')
train_labels.pop('MP4')
train_labels.pop('Te')
train_labels.pop('Hu')

test_labels = test_dataset.copy()
test_labels.pop('MP1')
test_labels.pop('MP2')
test_labels.pop('MP3')
test_labels.pop('MP4')
test_labels.pop('Te')
test_labels.pop('Hu')

train_dataset.pop('TeM1')
train_dataset.pop('TeC1')
train_dataset.pop('TeTD')
train_dataset.pop('TeD')
train_dataset.pop('TeTP1')
train_dataset.pop('TeP1')
train_dataset.pop('TeTP2')
train_dataset.pop('TeP2')
train_dataset.pop('TeTP3')
train_dataset.pop('TeP3')
train_dataset.pop('TeTP4')
train_dataset.pop('TeP4')
train_dataset

test_dataset.pop('TeM1')
test_dataset.pop('TeC1')
test_dataset.pop('TeTD')
test_dataset.pop('TeD')
test_dataset.pop('TeTP1')
test_dataset.pop('TeP1')
test_dataset.pop('TeTP2')
test_dataset.pop('TeP2')
test_dataset.pop('TeTP3')
test_dataset.pop('TeP3')
test_dataset.pop('TeTP4')
test_dataset.pop('TeP4')
test_dataset

def norm(x):
  return(x - train_stats['min']) /(train_stats['max']-train_stats['min'])

scaler = MinMaxScaler()
normed_dataset_orig = scaler.fit_transform(dataset_orig)

normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

"""Przygotowanie wyników które chcemy uzyskać do predykcji"""

predict_data = [1.8, 1.8, 1.8, 1.8, 23.9, 46]

normed_predict_data = norm(predict_data)
normed_predict_data = normed_predict_data.values

X = normed_train_data
y = train_labels

n_inputs, n_outputs = normed_train_data.shape[1], train_labels.shape[1]
model = MultiModel(n_inputs, n_outputs)
model.summary()

"""K-składowa walidacja modelu"""

k = 10
num_val_samples = len(normed_train_data) // k
num_epochs = 400
all_mae_histories = []

for i in range(k):
  print('processing fold #', i)
  val_data = normed_train_data[i * num_val_samples: (i+1) * num_val_samples]
  val_targets = train_labels[i * num_val_samples: (i+1) * num_val_samples]

  partial_train_data = np.concatenate([normed_train_data[:i * num_val_samples], normed_train_data[(i+1) * num_val_samples:]], axis = 0)
  partial_train_targets = np.concatenate([train_labels[:i * num_val_samples], train_labels[(i+1) * num_val_samples:]], axis = 0)
  n_inputs, n_outputs = partial_train_data.shape[1], partial_train_targets.shape[1]
  model = MultiModel1(n_inputs, n_outputs)
  history = model.fit(partial_train_data, partial_train_targets, validation_data = (val_data, val_targets), epochs=num_epochs, verbose=0, batch_size=3)

  mae_history = history.history['val_mae']
  print(mae_history)
  all_mae_histories.append(mae_history)

"""Obliczenie oraz wizualizacja średniego błędu bezwzględnego"""

average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for  i in range(num_epochs)]

plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel('Liczba epok')
plt.ylabel('Sredni blad bezwzgledny')
plt.show()

"""Odcięcie pierwszych k wystąpień"""

def smooth_curve(points, factor = 0.9):
  smoothed_points = []
  for point in points:
    if smoothed_points:
      previous = smoothed_points[-1]
      smoothed_points.append(previous * factor + point * (1 - factor))
    else:
      smoothed_points.append(point)
  return smoothed_points

smooth_mae_history = smooth_curve(average_mae_history[20:])

plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
plt.xlabel('Liczba epok')
plt.ylabel('Sredni blad bezwzgledny')
plt.show()

"""Ostateczne trenowanie modelu"""

n_inputs, n_outputs = normed_train_data.shape[1], train_labels.shape[1]
model = MultiModel(n_inputs, n_outputs)
model.fit(normed_train_data, train_labels, epochs = 400, batch_size = 3, verbose = 0)
for name, value in zip(model.metrics_names, model.evaluate(normed_test_data, test_labels.values)):
  print(f'{name:8}{value:.4f}')

"""Predykcja wymaganych wartości"""

newX = np.asarray([normed_predict_data])
print(newX.shape)
yhat = model.predict(newX)
yhat_round = np.around(yhat)
print('Predicted: %s' % yhat_round[0])

"""Predykcja wartości testowych"""

normed_test_predictions = model.predict(normed_test_data).flatten()
test_pred_round = np.around(normed_test_predictions)

out1, out2, out3, out4, out5, out6, out7, out8, out9, out10, out11, out12 = model.predict(normed_test_data)
normed_test_predictions = [out1, out2, out3, out4, out5, out6, out7, out8, out9, out10, out11, out12]
test_pred_round = np.around(normed_test_predictions).flatten()

"""Tabela porównująca dane"""

pred = pd.DataFrame(test_labels['TeM1'])
pred['Pred_TeM1'] = test_pred_round[0::12]
pred['TeC1'] = test_labels['TeC1']
pred['Pred_TeC1'] = test_pred_round[1::12]
pred['TeTD'] = test_labels['TeTD']
pred['Pred_TeTD'] = test_pred_round[2::12]
pred['TeD'] = test_labels['TeD']
pred['Pred_TeD'] = test_pred_round[3::12]
pred['TeTP1'] = test_labels['TeTP1']
pred['Pred_TeTP1'] = test_pred_round[4::12]
pred['TeP1'] = test_labels['TeP1']
pred['Pred_TeP1'] = test_pred_round[5::12]
pred['TeTP2'] = test_labels['TeTP2']
pred['Pred_TeTP2'] = test_pred_round[6::12]
pred['TeP2'] = test_labels['TeP1']
pred['Pred_TeP2'] = test_pred_round[7::12]
pred['TeTP3'] = test_labels['TeTP3']
pred['Pred_TeTP3'] = test_pred_round[8::12]
pred['TeP3'] = test_labels['TeP3']
pred['Pred_TeP3'] = test_pred_round[9::12]
pred['TeTP4'] = test_labels['TeTP4']
pred['Pred_TeTP4'] = test_pred_round[10::12]
pred['TeP4'] = test_labels['TeP4']
pred['Pred_TeP4'] = test_pred_round[11::12]
pred
pd.set_option('display.max_columns', None)
pred

"""Wykres porównania wartości"""

fig_pred = px.scatter(pred, 'TeTP2', 'Pred_TeP2')
fig_pred.add_trace(go.Scatter(x = [150,174], y = [150,174], mode='lines'))
fig_pred.show()

"""Zapis modelu"""

model.save('model121120loss14.h5')

"""Załadowanie przetrenowanego modelu"""

from tensorflow.keras.models import load_model

new_model = load_model('')
new_model.summary()

"""Sprawdzenie załadowanego modelu"""

for name, value in zip(new_model.metrics_names, new_model.evaluate(normed_test_data, test_labels.values)):
  print(f'{name:8}{value:.4f}')

newX = np.asarray([normed_predict_data])
print(newX.shape)
yhat = new_model.predict(newX)
yhat_round = np.around(yhat)
print('Predicted: %s' % yhat_round[0])

normed_test_predictions = new_model.predict(test_data).flatten()
normed_test_pred_round = np.around(normed_test_predictions)
