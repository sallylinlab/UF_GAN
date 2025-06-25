import tensorflow as tf

class MultiFeatureLoss(tf.keras.losses.Loss):
    def __init__(self,
                 reduction=tf.keras.losses.Reduction.AUTO,
                 name='FeatureLoss'):
        super().__init__(reduction=reduction, name=name)
        self.mse_func = tf.keras.losses.MeanSquaredError()

    def call(self, real, fake, weight=1):
        result = 0.0
        for r, f in zip(real, fake):
            result = result + (weight * self.mse_func(r, f))

        return result


# class for Adversarial loss function
class AdversarialLoss(tf.keras.losses.Loss):
    def __init__(self,
                 reduction=tf.keras.losses.Reduction.AUTO,
                 name='AdversarialLoss'):
        super().__init__(reduction=reduction, name=name)

    def call(self, logits_in, labels_in):
        labels_in = tf.convert_to_tensor(labels_in)
        logits_in = tf.cast(logits_in, labels_in.dtype)
        # Loss 4: FEATURE Loss
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_in, labels=labels_in))


# class for Charbonnier loss function
class CharbonnierLoss(tf.keras.losses.Loss):
    def __init__(self,
                 reduction=tf.keras.losses.Reduction.AUTO,
                 name='CharbonnierLoss'):
        super().__init__(reduction=reduction, name=name)

    def call(self, real, fake):
        epsilon = 1e-3
        # error = real - fake
        error = tf.subtract(real, fake)
        result = tf.math.sqrt(tf.math.square(error) + tf.math.square(epsilon))

        return tf.reduce_mean(result)
