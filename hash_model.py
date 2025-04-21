import keras
from keras import backend as K
from tensorflow.keras.backend import mean, square, sum
import numpy as np
from keras.layers import BatchNormalization
from keras.layers import Conv2D, Dense, Input, add, Activation, GlobalAveragePooling2D, Flatten, Reshape, UpSampling2D, Conv2DTranspose, MaxPooling2D
from keras.initializers import he_normal
from keras import regularizers
from keras import losses
import tensorflow as tf

weight_decay = 0.0005

def residual_block(x, shape, filter_type=''):
    output_filter_num = shape[1]
    if filter_type == 'increase':   # set the stride to (2, 2) is just like pooling
        first_stride = (2, 2)
    elif filter_type == 'decrease':
        x = UpSampling2D()(x)         # if filter is decrease, we Unsample the x first
        first_stride = (1, 1)
    else:
        first_stride = (1, 1)

    pre_bn = BatchNormalization()(x)
    pre_relu = Activation('relu')(pre_bn)

    conv_1 = Conv2D(output_filter_num,
                    kernel_size=(3, 3),
                    strides=first_stride,    # if 'increase', change the feature map size here (pooling)
                    padding='same',
                    kernel_initializer=he_normal(),
                    kernel_regularizer=regularizers.l2(weight_decay)
                    )(pre_relu)
    bn_1 = BatchNormalization()(conv_1)
    relu1 = Activation('relu')(bn_1)
    conv_2 = Conv2D(output_filter_num,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    padding='same',
                    kernel_initializer=he_normal(),
                    kernel_regularizer=regularizers.l2(weight_decay)
                    )(relu1)
    if filter_type == 'increase': # change the image size and channel from x to block
        projection = Conv2D(output_filter_num,
                            kernel_size=(1, 1),
                            strides=(2, 2),
                            padding='same',
                            kernel_initializer=he_normal(),
                            kernel_regularizer=regularizers.l2(weight_decay)
                            )(x)
        block = add([conv_2, projection])
    elif filter_type == 'decrease':
        projection = Conv2D(output_filter_num,
                            kernel_size=(1,1),
                            strides=(1, 1),
                            kernel_initializer=he_normal(),
                            kernel_regularizer=regularizers.l2(weight_decay)
                            )(x)
        block = add([conv_2, projection])
    else:
        block = add([conv_2, x])
    return block


# abstract class for hash model
# you have to define the children class to inherate the __init__ and net_loss function,
# and name your hash_layer to 'hash_x'

class HashModel:
    def __init__(self, img_rows, img_cols, img_channels, num_classes, stack_num, hash_bits):
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.img_channels = img_channels
        self.num_classes = num_classes
        self.stack_num = stack_num
        self.hash_bits = hash_bits

    # you have to overite this function for the inherrent class
    def net_loss(self, y_true, y_pred):
        return 0

class HashSupervisedAutoEncoderModel(HashModel):
    def __init__(self, img_rows, img_cols, img_channels, num_classes, stack_num, hash_bits, alpha, beta, gamma):
        # Initialize base HashModel
        HashModel.__init__(self, img_rows, img_cols, img_channels, num_classes, stack_num, hash_bits)
        
        # Hyperparameters for loss weighting
        self.alpha = alpha  # weight for classification loss
        self.beta = beta    # weight for reconstruction loss
        self.gamma = gamma  # weight for quantization loss

        # Define the image input
        self.img_input = Input(shape=(self.img_rows, self.img_cols, self.img_channels), name="img_input")

        # ========================
        # Encoder Subnet
        # ========================
        # Initial convolution
        x = Conv2D(filters=16,
                   kernel_size=(3, 3),
                   strides=(1, 1),
                   padding='same',
                   kernel_initializer=he_normal(),
                   kernel_regularizer=regularizers.l2(weight_decay),
                   )(self.img_input)

        # Residual blocks at different feature scales
        for _ in range(0, self.stack_num):
            x = residual_block(x, [16, 16])  # Keep feature size

        x = residual_block(x, [16, 32], filter_type='increase')  # Downsample + increase depth
        for _ in range(1, self.stack_num):
            x = residual_block(x, [16, 32])  # More processing at 32-channel scale

        x = residual_block(x, [32, 64], filter_type='increase')  # Further downsample + increase depth
        for _ in range(1, self.stack_num):
            x = residual_block(x, [32, 64])  # More processing at 64-channel scale

        # Final feature normalization and activation
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # Store shape for later restoration in decoder
        shape_restore = x.shape[1:4]
        units_restore = shape_restore[0] * shape_restore[1] * shape_restore[2]

        # Flatten to vector
        x = Flatten()(x)

        # ========================
        # Hash Layer
        # ========================
        # Learnable hash code (sigmoid bounded binary-like output)
        self.hash_x = Dense(hash_bits, activation='sigmoid', kernel_initializer=he_normal(),
                            kernel_regularizer=regularizers.l2(weight_decay), name="hash_x")(x)

        # ========================
        # Decoder Subnet
        # ========================
        # Reconstruct latent feature map from hash
        x = Dense(units_restore, activation='relu', kernel_initializer=he_normal(),
                  kernel_regularizer=regularizers.l2(weight_decay))(self.hash_x)
        x = Reshape((shape_restore[0], shape_restore[1], shape_restore[2]))(x)

        # Upsample and decode using residual blocks
        for _ in range(1, self.stack_num):
            x = residual_block(x, [64, 64])  # Keep size at 64 channels
        x = residual_block(x, [64, 32], filter_type='decrease')  # Upsample and reduce channels

        for _ in range(1, self.stack_num):
            x = residual_block(x, [32, 32])
        x = residual_block(x, [32, 16], filter_type='decrease')

        for _ in range(0, self.stack_num):
            x = residual_block(x, [16, 16])  # Final refinement at original feature depth

        # Final normalization and activation
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # Output reconstructed image (same shape as input)
        self.y_decoded = Conv2D(filters=self.img_channels,
                        activation='sigmoid',
                        kernel_size=(3, 3),
                        strides=(1, 1),
                        padding='same',
                        kernel_initializer=he_normal(),
                        kernel_regularizer=regularizers.l2(weight_decay),
                        name='y_decoded')(x)

        # ========================
        # Supervisory Subnet
        # ========================
        # Class prediction from hash code
        self.y_predict = Dense(self.num_classes, activation='softmax', kernel_initializer=he_normal(),
                               kernel_regularizer=regularizers.l2(weight_decay), name='y_predict')(self.hash_x)

    # Required import for loss computation
    from tensorflow.keras.backend import mean, square, sum


def net_loss(self, y_true, y_pred):
    supervised_loss = 0  # losses.categorical_crossentropy(y_true, y_pred)
    binary_loss = - mean(square(self.hash_x - 0.5))  # Compatible with Keras tensors
    balance_loss = sum(square(mean(self.hash_x, axis=0) - 0.5))  # Compatible with Keras tensors
    decoded_loss = mean(square(y_true - y_pred))  # Compatible with Keras tensors

    return supervised_loss + self.alpha * binary_loss + self.beta * balance_loss + self.gamma * decoded_loss
