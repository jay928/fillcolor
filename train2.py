import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, Conv2DTranspose, UpSampling2D, Reshape, Input
from keras.models import Model

SOURCE_PATH = '/Users/jwp928/Documents/deeplearning/data/fillcolor/'
FILTERS, TARGET_SIZE, BATCH_SIZE, EPOCHS = 32, 256, 32, 10

x_data = np.load(SOURCE_PATH + 'x.npy', allow_pickle=True)
y_data = np.load(SOURCE_PATH + 'y.npy', allow_pickle=True)
print(x_data.shape)
print(y_data.shape)

x_data = np.reshape(x_data, (-1, 256, 256, 1))
print(x_data.shape)

input = Input(shape=(256, 256, 1), dtype='float32')
conv2d1 = Conv2D(filters=FILTERS,
                 kernel_size=(2, 2),
                 activation='relu',
                 input_shape=x_data.shape,
                 padding='same',
                 strides=1)(input)
maxp1 = MaxPooling2D(pool_size=(2, 2))(conv2d1)
conv2d2 = Conv2D(filters=FILTERS,
                 kernel_size=(2, 2),
                 activation='relu',
                 padding='same',
                 strides=1)(maxp1)
maxp2 = MaxPooling2D(pool_size=(2, 2))(conv2d2)

conv2d3 = Conv2D(filters=FILTERS,
                 kernel_size=(2, 2),
                 activation='relu',
                 padding='same',
                 strides=1)(maxp2)

upsam1 = UpSampling2D(size=(2, 2))(conv2d3)

conv2d4 = Conv2D(filters=FILTERS,
                 kernel_size=(2, 2),
                 activation='relu',
                 padding='same',
                 strides=1)(upsam1)

upsam2 = UpSampling2D(size=(2, 2))(conv2d4)

conv2d5 = Conv2D(filters=2,
                 kernel_size=(2, 2),
                 activation='sigmoid',
                 padding='same',
                 strides=1)(upsam2)

# dense1 = Dense(32, activation='relu')(maxp2)
# dense2 = Dense(32, activation='relu')(dense1)
output = Reshape((-1, 256, 256))(conv2d5)

model = Model(inputs=input, outputs=output)

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

model.fit(x_data, y_data, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1, shuffle=True, validation_split=0.1)

print("TRAINING COMPLETED!")