from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU
from keras.layers.convolutional import Convolution2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from get_data import *

def get_nvidia_model():
  """An implementation of NVIDIA end to end model using keras"""
  ch, row, col = 3, 64, 64
  dp = 0.5
  model = Sequential()
  model.add(Lambda(lambda x: x/127.5 - 1.,
            input_shape=(row, col, ch),
            output_shape=(row, col, ch), name='normalization'))
  model.add(Convolution2D(24, 3, 3, init='he_normal', subsample=(2, 2), border_mode="valid", name='conv2d_1'))
  model.add(ELU(name='ELU_1'))
  model.add(Dropout(dp, name= 'dropout1'))
  model.add(Convolution2D(36, 3, 3, init='he_normal', subsample=(2, 2), border_mode="valid", name='conv2d_2'))
  model.add(ELU(name='ELU_2'))
  model.add(Dropout(dp, name= 'dropout2'))
  model.add(Convolution2D(48, 3, 3, init='he_normal', subsample=(2, 2), border_mode="valid", name='conv2d_3'))
  model.add(ELU(name='ELU_3'))
  model.add(Dropout(dp, name= 'dropout3'))
  model.add(Convolution2D(64, 3, 3, init='he_normal', subsample=(1, 1), border_mode="valid", name='conv2d_4'))
  model.add(ELU(name='ELU_4'))
  model.add(Dropout(dp, name= 'dropout4'))
  model.add(Convolution2D(64, 3, 3, init='he_normal', subsample=(1, 1), border_mode="valid", name='conv2d_5'))
  model.add(ELU(name='ELU_5'))
  model.add(Dropout(dp, name= 'dropout5'))
  model.add(Flatten(name='flatten1'))
  #model.add(Dense(576, init='he_normal', activation=ELU))
  #model.add(Dropout(dp))
  dp = 0.2
  model.add(Dense(100, init='he_normal', name='fc1'))
  model.add(ELU(name='ELU_6'))
  model.add(Dropout(dp, name= 'dropout6'))
  model.add(Dense(50, init='he_normal', name='fc2'))
  model.add(ELU(name='ELU_7'))
  model.add(Dropout(dp, name= 'dropout7'))
  model.add(Dense(10, init='he_normal', name='fc3'))
  model.add(ELU(name='ELU_8'))
  model.add(Dropout(dp, name= 'dropout8'))
  model.add(Dense(1, init='he_normal', name='output_fc'))

  adam = Adam(lr = 1e-4)
  model.compile(adam, loss="mse")

  return model

def train_model(model, df_train, df_valid, batch_size, nb_epoch, train_samples_per_epoch, val_samples_per_epoch):
    """Function to train a keras model and store the model and weights"""
    # Serialize model to JSON and store it
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # create checkpoints to write out weights after every epoch
    checkpoint = ModelCheckpoint(filepath="model.{epoch:02d}.h5", verbose=1)
    history = model.fit_generator(aug_data_generator(df_train, batch_size), samples_per_epoch=train_samples_per_epoch, nb_epoch = nb_epoch, \
                                  validation_data=valid_data_generator(df_valid, batch_size), nb_val_samples=val_samples_per_epoch,
                                  callbacks=[checkpoint])  # num_train, num_valid
    return history

def test_nvidia_model(model):
    """Simple unit tests for the NVIDIA model"""
    assert model.layers[0].input_shape == (None, 64, 64, 3), 'First layer input shape is wrong, it should be (64, 64, 3)'
    assert len(model.layers) == 27, 'Number of layers should be 27!'
    assert model.layers[len(model.layers)-1].output_shape == (None, 1), 'Ouput layer output is wrong, it should be (1)'
    print('Tests passed.')