


# #############################
# # Choose your model here    #
# #############################
# model = "CNN_img"
# #############################


# if model== "CNN":
#     import G_Main.AircraftClassification.exp_CNN as CNN
#     CNN.__main__()

# if model== "CNN_img":
#     import G_Main.AircraftClassification.exp_CNN_img as CNN_img
#     CNN_img.__main__()

# if model== "LSTM_img":
#     import G_Main.AircraftClassification.exp_LSTM_img as LSTM_img
#     LSTM_img.__main__()

# elif model== "LSTM":
#     import G_Main.AircraftClassification.exp_LSTM as LSTM
#     LSTM.__main__()

# elif model== "Transformer":
#     import G_Main.AircraftClassification.exp_Transformer as Transformer
#     Transformer.__main__()

# elif model== "Transformer_img":
#     import G_Main.AircraftClassification.exp_Transformer_img as Transformer
#     Transformer.__main__()

# elif model== "Reservoir":
#     import G_Main.AircraftClassification.exp_Reservoir as Reservoir
#     Reservoir.__main__()




import tensorflow as tf
import time

# # print gpu
print("Num GPUs Available: ", tf.config.experimental.list_physical_devices('GPU'))


# # run on CPU
# # tf.config.set_visible_devices([], 'GPU')

# # limitate gpu usage
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#   # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
#   try:
#     tf.config.set_logical_device_configuration(
#         gpus[0],
#         [tf.config.LogicalDeviceConfiguration(memory_limit=1024*12)])
#     logical_gpus = tf.config.list_logical_devices('GPU')
#     print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#   except RuntimeError as e:
#     # Virtual devices must be set before GPUs have been initialized
#     print(e)




mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape((-1, 28, 28, 1)).astype('float32') / 255.0
x_test = x_test.reshape((-1, 28, 28, 1)).astype('float32') / 255.0

y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)


# conv 2d model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(100, activation='relu', kernel_initializer='he_uniform'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))
# compile model
opt = tf.keras.optimizers.SGD(lr=0.01, momentum=0.9)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])




start = time.time()


model.fit(
    x_train, y_train,
    epochs=10, batch_size=32,
    validation_data=(x_test, y_test),
)

print("Time taken: ", time.time() - start)