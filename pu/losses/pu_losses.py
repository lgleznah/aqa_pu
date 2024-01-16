import tensorflow as tf

def create_nn_pu_loss(positive_prior, loss_fn):
    def nn_pu_loss(y_true, y_pred):
        positive_examples = tf.squeeze(tf.gather(y_pred, tf.where(y_true == 1)))
        unlabeled_examples = tf.squeeze(tf.gather(y_pred, tf.where(y_true == 0)))

        positive_positive_risk = positive_prior * tf.reduce_mean(loss_fn(tf.ones_like(positive_examples), positive_examples))
        unlabeled_negative_risk = tf.reduce_mean(loss_fn(tf.zeros_like(unlabeled_examples), unlabeled_examples))
        positive_negative_risk = positive_prior * tf.reduce_mean(loss_fn(tf.zeros_like(positive_examples), positive_examples))

        loss = positive_positive_risk + tf.math.maximum(0.0, unlabeled_negative_risk - positive_negative_risk)

        return loss
    
    return nn_pu_loss