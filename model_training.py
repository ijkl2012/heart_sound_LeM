import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2


# ========= SAFE frontend: Learnable Mel with Conv-STFT =========
class LearnableMel(layers.Layer):
    """
    TF implementation of STFT: use conv1d with fixed cos/sin kernels to compute
    the power spectrum, avoiding tf.signal.frame/stft.
    Parameters: N=40, win=512, hop=256 (2 kHz). Softplus(theta) enforces non-negativity.
    Output shape: (B, T, N, 1).
    """

    def __init__(self, mel_bands=40, win=512, hop=256, eps=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.mel_bands = int(mel_bands)
        self.N = self.mel_bands
        self.win = int(win)
        self.hop = int(hop)
        self.eps_value = eps  # keep raw value for serialization
        self.eps = tf.constant(eps, tf.float32)

    def build(self, input_shape):
        # Number of frequency bins K
        self.K = self.win // 2 + 1

        # Generate Hann window
        hann = tf.signal.hann_window(self.win, periodic=True, dtype=tf.float32)  # (win,)
        hann = hann[:, None]  # (win, 1)

        # Generate DFT bases
        PI = tf.constant(3.141592653589793, tf.float32)
        n = tf.cast(tf.range(self.win), tf.float32)[:, None]  # (win, 1)
        k = tf.cast(tf.range(self.K), tf.float32)[None, :]    # (1, K)
        ang = 2.0 * PI * n * k / float(self.win)              # (win, K)

        basis_cos = hann * tf.cos(ang)  # (win, K)
        basis_sin = hann * tf.sin(ang)  # (win, K)

        # conv1d kernel shape: [filter_width, in_channels, out_channels] = (win, 1, K)
        self.kernel_cos = tf.Variable(
            tf.expand_dims(basis_cos, axis=1), trainable=False, name="dft_cos"
        )
        self.kernel_sin = tf.Variable(
            tf.expand_dims(basis_sin, axis=1), trainable=False, name="dft_sin"
        )

        # Learnable Mel filters: theta → A = Softplus(theta) with shape (N, K)
        self.theta = self.add_weight(
            name='theta', shape=(self.N, self.K),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
            trainable=True
        )

    @tf.function
    def _pad_to_valid_stride(self, x3):
        """
        Compute the right padding length on the time dimension so that
        conv1d(stride=hop, padding='VALID') fully covers the last frame.
        x3: (B, L, 1)
        """
        L = tf.shape(x3)[1]
        win = tf.constant(self.win, tf.int32)
        hop = tf.constant(self.hop, tf.int32)

        # frames = floor(max(L - win, 0) / hop) + 1
        frames = tf.math.floordiv(tf.maximum(L - win, 0), hop) + 1
        last_end = frames * hop + win  # end position of the last window
        pad_right = tf.maximum(0, last_end - L)  # required right-side padding
        paddings = tf.stack([[0, 0], [0, pad_right], [0, 0]])
        return tf.pad(x3, paddings), pad_right

    def call(self, x):
        # x: (B, L) -> (B, L, 1)
        x3 = tf.expand_dims(x, axis=-1)

        # Ensure the last window is complete under stride=hop and padding='VALID'
        x3, _ = self._pad_to_valid_stride(x3)  # (B, L_pad, 1)

        # Convolve to obtain real/imag parts per frequency bin (orthogonal bases)
        # Output: (B, T, K), where T is determined by hop
        real = tf.nn.conv1d(x3, self.kernel_cos, stride=self.hop, padding='VALID')
        imag = tf.nn.conv1d(x3, self.kernel_sin, stride=self.hop, padding='VALID')

        # Power spectrum
        P = tf.square(real) + tf.square(imag)  # (B, T, K)

        # Learnable Mel filtering: A = Softplus(theta); (N, K) -> (K, N)
        A_T = tf.transpose(tf.nn.softplus(self.theta), [1, 0])  # (K, N)
        M = tf.matmul(P, A_T)  # (B, T, N)
        E = tf.math.log(M + self.eps)  # (B, T, N)
        return tf.expand_dims(E, axis=-1)  # (B, T, N, 1)

    def get_config(self):
        """Add get_config to support model serialization."""
        config = super(LearnableMel, self).get_config()
        config.update({
            'mel_bands': self.mel_bands,
            'win': self.win,
            'hop': self.hop,
            'eps': self.eps_value
        })
        return config


class BandSE(layers.Layer):
    """Squeeze & Excitation along the Mel-band channel N (input shape (B, T, N, C))."""

    def __init__(self, channels_n: int, reduction: int = 8, **kwargs):
        super().__init__(**kwargs)
        self.channels_n = channels_n
        self.reduction = reduction
        r = max(1, channels_n // reduction)
        self.avg = layers.Lambda(lambda z: tf.reduce_mean(z, axis=[1, 3], keepdims=False))  # (B, N)
        self.fc1 = layers.Dense(r, activation='relu')
        self.fc2 = layers.Dense(channels_n, activation='sigmoid')
        self.reshape = layers.Reshape((1, channels_n, 1))
        self.mul = layers.Multiply()

    def call(self, x):
        z = self.avg(x)
        z = self.fc1(z)
        z = self.fc2(z)
        z = self.reshape(z)
        return self.mul([x, z])

    def get_config(self):
        """Add get_config to support model serialization."""
        config = super(BandSE, self).get_config()
        config.update({
            'channels_n': self.channels_n,
            'reduction': self.reduction
        })
        return config


class DotProductMHA(layers.Layer):
    def __init__(self, num_heads=4, dk=32, dv=32, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = int(num_heads)
        self.dk = int(dk)
        self.dv = int(dv)
        self.h = self.num_heads  # keep backward compatibility

    def build(self, input_shape):
        d_in = int(input_shape[-1])
        init = tf.keras.initializers.GlorotUniform()
        self.wq = self.add_weight(shape=(d_in, self.h * self.dk), initializer=init, name='wq')
        self.wk = self.add_weight(shape=(d_in, self.h * self.dk), initializer=init, name='wk')
        self.wv = self.add_weight(shape=(d_in, self.h * self.dv), initializer=init, name='wv')
        self.wo = self.add_weight(shape=(self.h * self.dv, d_in), initializer=init, name='wo')

    def call(self, x):
        # x: (B, T, D)
        q = tf.tensordot(x, self.wq, axes=1)
        k = tf.tensordot(x, self.wk, axes=1)
        v = tf.tensordot(x, self.wv, axes=1)
        B = tf.shape(x)[0]
        T = tf.shape(x)[1]
        q = tf.reshape(q, (B, T, self.h, self.dk))
        k = tf.reshape(k, (B, T, self.h, self.dk))
        v = tf.reshape(v, (B, T, self.h, self.dv))
        q = tf.transpose(q, [0, 2, 1, 3])  # (B, H, T, dk)
        k = tf.transpose(k, [0, 2, 1, 3])  # (B, H, T, dk)
        v = tf.transpose(v, [0, 2, 1, 3])  # (B, H, T, dv)
        scale = tf.math.rsqrt(tf.cast(self.dk, tf.float32))
        attn = tf.nn.softmax(tf.matmul(q, k, transpose_b=True) * scale, axis=-1)  # (B, H, T, T)
        o = tf.matmul(attn, v)  # (B, H, T, dv)
        o = tf.transpose(o, [0, 2, 1, 3])  # (B, T, H, dv)
        o = tf.reshape(o, (B, T, self.h * self.dv))
        return tf.tensordot(o, self.wo, axes=1)  # (B, T, D)

    def get_config(self):
        """Add get_config to support model serialization."""
        config = super(DotProductMHA, self).get_config()
        config.update({
            'num_heads': self.num_heads,
            'dk': self.dk,
            'dv': self.dv
        })
        return config


# ========= Multi-branch residual (3×3 & 5×5) =========
def conv_bn_relu(x, filters, k, wd):
    x = layers.Conv2D(filters, k, padding='same', use_bias=False, kernel_regularizer=l2(wd))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x


def mbr_block(x, filters, wd, dropout=0.3):
    b1 = conv_bn_relu(x, filters, (3, 3), wd)
    b2 = conv_bn_relu(x, filters, (5, 5), wd)
    out = layers.Concatenate()([b1, b2])
    out = conv_bn_relu(out, filters, (1, 1), wd)
    skip = conv_bn_relu(x, filters, (1, 1), wd) if x.shape[-1] != filters else x
    out = layers.Add()([out, skip])
    return layers.Dropout(dropout)(out)


# ========= LeM-MARNet =========
def build_model(input_shape, num_classes=1):
    """
    Args:
      - Raw waveform at 2 kHz, input shape (L,)
      - Learnable Mel: N=40, win=512, hop=256
      - Trunk: channels 32/64/128 with multi-branch residual (3×3 & 5×5); downsample along time
      - Attention: H=4, d_k=32
      - Head: Dense(128) -> Dropout(0.3) -> Sigmoid/Softmax
      - Training: Adam(lr=8e-4) + Binary Cross-Entropy; L2 weight decay = 1e-2
    """
    wd = 1e-2
    dr = 0.3

    inp = layers.Input(shape=input_shape, name="wave")  # (L,)
    x = LearnableMel(mel_bands=40, win=512, hop=256)(inp)  # (B, T, N, 1)
    x = BandSE(channels_n=40, reduction=8)(x)  # (B, T, N, 1)

    for i, f in enumerate([32, 64, 128]):
        if i > 0:
            x = layers.AveragePooling2D(pool_size=(2, 1))(x)  # downsample time axis T only
        x = mbr_block(x, filters=f, wd=wd, dropout=dr)

    # Average over N to get (B, T, C); then apply MHA and global pooling
    seq = tf.reduce_mean(x, axis=2)  # (B, T, C)
    seq = DotProductMHA(num_heads=4, dk=32, dv=32)(seq)  # (B, T, C)
    seq = layers.LayerNormalization()(seq)
    seq = layers.GlobalAveragePooling1D()(seq)  # (B, C)

    h = layers.Dense(128, activation='relu', kernel_regularizer=l2(wd))(seq)
    h = layers.Dropout(dr)(h)

    if num_classes == 1:
        out = layers.Dense(1, activation='sigmoid', kernel_regularizer=l2(wd))(h)
        loss = 'binary_crossentropy'
    else:
        out = layers.Dense(num_classes, activation='softmax', kernel_regularizer=l2(wd))(h)
        loss = 'sparse_categorical_crossentropy'

    model = Model(inputs=inp, outputs=out, name="LeM_MARNet")
    model.compile(optimizer=Adam(learning_rate=8e-4), loss=loss, metrics=['accuracy'])
    return model


