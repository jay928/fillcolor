import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, Conv2DTranspose, UpSampling2D, Reshape, Input

SOURCE_PATH = '/Users/jwp928/Documents/deeplearning/data/fillcolor/'
FILTERS, TARGET_SIZE, BATCH_SIZE, EPOCHS = 32, 256, 32, 10

x_data = np.load(SOURCE_PATH + 'x.npy', allow_pickle=True)
y_data = np.load(SOURCE_PATH + 'y.npy', allow_pickle=True)

print(x_data.shape)
print(y_data.shape)

model = Sequential()
model.add(Conv2D(filters=FILTERS,
                 kernel_size=(2, 2),
                 activation='relu',
                 input_shape=x_data.shape,
                 padding='same',
                 strides=1))
print(model.output_shape)
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
print(model.output_shape)

model.add(Conv2D(filters=FILTERS,
                 kernel_size=(2, 2),
                 activation='relu',
                 padding='same',
                 strides=1))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=FILTERS,
                 kernel_size=(2, 2),
                 activation='relu',
                 padding='same',
                 strides=1))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=FILTERS,
                 kernel_size=(2, 2),
                 activation='relu',
                 padding='same',
                 strides=1))
print(model.output_shape)

model.add(UpSampling2D(size=(2, 2)))
print(model.output_shape)

model.add(Conv2D(filters=FILTERS,
                 kernel_size=(2, 2),
                 activation='relu',
                 padding='same',
                 strides=1))
print(model.output_shape)

model.add(UpSampling2D(size=(2, 2)))
print(model.output_shape)

model.add(Conv2D(filters=2,
                 kernel_size=(3, 3),
                 activation='sigmoid',
                 padding='same'))

print(model.output_shape)#
# model.add(Conv2D(filters=FILTERS,
#                  kernel_size=(2, 2),
#                  activation='relu',
#                  padding='same',
#                  strides=1))
# print(model.output_shape)
#
# model.add(UpSampling2D(size=(2, 2)))
# print(model.output_shape)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_data, y_data, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1, shuffle=True, validation_split=0.1)

print("TRAINING COMPLETED!")