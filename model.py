from keras.layers import (
    Activation, Add, Dense, Flatten, Input, Multiply, BatchNormalization)
from keras.models import Model
from groupnorm import GroupNormalization
from keras.optimizers import Adam
from tensorflow.keras.layers import LayerNormalization, Conv1D
import tensorflow as tf


def WaveNetResidualConv1D(num_channel, num_filters, kernel_size, dilation_rate):
    """ Function that creates a residual block for the WaveNet with gated
        activation units, skip connections and residual output, as described
        in Sections 2.3 and 2.4 of the paper [1].
        Args:
            num_filters (int): Number of filters used for convolution.
            kernel_size (int): The size of the convolution.
            dilation_rate (int): The dilation rate for the dilated convolution.
        Returns:
            A layer wrapper compatible to the Keras functional API.
        See:
            [1] Oord, Aaron van den, et al. "Wavenet: A generative model for
                raw audio." arXiv preprint arXiv:1609.03499 (2016).
    """
    def build_residual_block(l_input):
        # Gated activation.
        l_norm = GroupNormalization(groups=1)(l_input)
        l_sigmoid_conv1d = tf.keras.layers.Conv1D(
            num_filters, kernel_size, dilation_rate=dilation_rate,
            padding="causal", activation="sigmoid", groups=4)(l_norm)
        l_tanh_conv1d = tf.keras.layers.Conv1D(
            num_filters, kernel_size, dilation_rate=dilation_rate,
            padding="causal", activation="relu", groups=4)(l_input)
        l_mul = Multiply()([l_sigmoid_conv1d, l_tanh_conv1d])

        # Branches out to skip unit and residual output.
        l_skip = GroupNormalization(groups=1)(l_mul)
        l_skip_connection = tf.keras.layers.Conv1D(num_channel, 1, padding="causal")(l_skip)
        l_res = GroupNormalization(groups=1)(l_mul)
        l_residual = tf.keras.layers.Conv1D(num_filters, 1, padding="causal")(l_res)

        l_residual = Add()([l_input, l_residual])
        return l_residual, l_skip_connection
    return build_residual_block


def build_wavenet_model(input_size, num_channel, num_filters, kernel_size,
                        num_residual_blocks):
    """ Returns an implementation of WaveNet, as described in Section 2
        of the paper [1].
        Args:
            input_size (int): The size of the waveform the network will
                consider as input.
            num_filters (int): Number of filters used for convolution.
            kernel_size (int): The size of the convolution.
            num_residual_blocks (int): How many residual blocks to generate
                between input and output. Residual block i will have a dilation
                rate of 2^(i+1), i starting from zero.
        Returns:
            A Keras model representing the WaveNet.
        See:
            [1] Oord, Aaron van den, et al. "Wavenet: A generative model for
                raw audio." arXiv preprint arXiv:1609.03499 (2016).
    """
    l_input = Input(batch_shape=(None, input_size, num_channel))
    l_norm = GroupNormalization(groups=1)(l_input)
    l_stack_conv1d = tf.keras.layers.Conv1D(num_filters,
                                            1,
                                            padding="causal",
                                            groups=4,
                                            )(l_norm)
    l_skip_connections = []
    for i in range(num_residual_blocks):
        l_stack_conv1d, l_skip_connection = WaveNetResidualConv1D(
            num_channel, num_filters, kernel_size, 2 ** i)(l_stack_conv1d)
        l_skip_connections.append(l_skip_connection)

    # one causal 1d conv added for FREAK model
    l_skip_connection = tf.keras.layers.Conv1D(num_channel, kernel_size,
                                               activation="relu",
                                               padding="causal",
                                               groups=4,
                                               dilation_rate=8)(l_stack_conv1d)

    l_skip_connections.append(l_skip_connection)

    l_sum = Add()(l_skip_connections)
    relu = Activation("relu")(l_sum)
    relu = GroupNormalization(groups=1)(relu)
    l1_conv1d = tf.keras.layers.Conv1D(num_channel, 1, activation="relu", padding="causal", groups=4)(relu)
    l1_conv1d = GroupNormalization(groups=1)(l1_conv1d)
    l2_conv1d = tf.keras.layers.Conv1D(num_channel, 1, padding="causal", groups=4)(l1_conv1d)
    # l_flatten = Flatten()(l2_conv1d)
    # l_output = Dense(128)(l_flatten)
    l2_conv1d = l2_conv1d[:, 30:-1, :]
    model = Model(inputs=[l_input], outputs=[l2_conv1d])
    model.compile(
        loss="mse", optimizer=Adam(beta_1=0.85)
    )
    return model
