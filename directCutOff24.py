import os
import random
import tensorflow as tf
from policyGradient import *
from logger import logger
keras = tf.keras
l = keras.layers
batch_size = 128
num_classes = 10
epochs = 10
checkpoint_path = "training_1/cp.ckpt"

b = 0.5  # policyGradient 损失函数超参数

def build_functional_model(input_shape):
  inp = tf.keras.Input(shape=input_shape)
  x = l.Conv2D(32, 5, padding='same', activation='relu')(inp)
  x = l.MaxPooling2D((2, 2), (2, 2), padding='same')(x)
  x = l.BatchNormalization()(x)
  x = l.Conv2D(64, 5, padding='same', activation='relu')(x)
  x = l.MaxPooling2D((2, 2), (2, 2), padding='same')(x)
  x = l.Flatten()(x)
  x = l.Dense(1024, activation='relu')(x)
  x = l.Dropout(0.4)(x)
  out = l.Dense(num_classes, activation='softmax')(x)

  return tf.keras.models.Model([inp], [out])



if __name__=="__main__":
    img_rows, img_cols = 28, 28

    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    if tf.keras.backend.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)
    model = build_functional_model(input_shape)
    saved_model_dir = '/tmp/saved_model'

    model.compile(
        loss=tf.keras.losses.categorical_crossentropy,
        optimizer='adam',
        metrics=['accuracy'])

    # Print the model summary.
    model.summary()
    #如果不存在训练好的模型，训练一个模型并保存
    if not os.path.exists(saved_model_dir):
        model.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(x_test, y_test))
        score = model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        os.makedirs(saved_model_dir)
        print('Saving model to: ', saved_model_dir)
        #保存模型
        tf.keras.models.save_model(model, saved_model_dir, save_format='tf')
    #加载已有模型
    net = tf.keras.models.load_model(saved_model_dir)
    score = net.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    origin_acc=score[1]
    all_weights_of_whole_CNN = net.get_weights()

    layer=24#剪枝的层数
    Change_a_little_dens3 = all_weights_of_whole_CNN[layer].copy()
    # 取绝对值
    abs_Change_a_little_dens3_temp = abs(Change_a_little_dens3)
    # 拉平张量
    abs_observation = abs_Change_a_little_dens3_temp.reshape((1, -1))[0]
    Change_a_little_dens3_temp=Change_a_little_dens3.reshape((1, -1))[0]
    dense_length=len(Change_a_little_dens3_temp)
    pruning_rate=0.9
    index_of_Descending_Order = np.argsort(-abs(abs_observation))
    for j in range(0, len(index_of_Descending_Order)):
        if (j >= int(100 * (1 - pruning_rate))):
            Change_a_little_dens3_temp[index_of_Descending_Order[j]] = 0




    Change_a_little_dens3=Change_a_little_dens3_temp.reshape(Change_a_little_dens3.shape)
    all_weights_of_whole_CNN[layer] = Change_a_little_dens3
    net.set_weights(all_weights_of_whole_CNN)
    score = net.evaluate(x_test[:1000], y_test[:1000], verbose=0)

            #根据公式获取损失函数的一部分

    logger.info("cnn模型{}层直接剪情况 参数个数:{} pruning rate:{:.4} acc:{:.2}".format(layer,dense_length,pruning_rate,score[1]))

