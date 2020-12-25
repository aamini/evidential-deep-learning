import tensorflow as tf
import numpy as np


def Dirichlet_SOS(y, alpha, t):
    def KL(alpha):
        beta=tf.constant(np.ones((1,alpha.shape[1])),dtype=tf.float32)
        S_alpha = tf.reduce_sum(alpha,axis=1,keepdims=True)
        S_beta = tf.reduce_sum(beta,axis=1,keepdims=True)
        lnB = tf.math.lgamma(S_alpha) - tf.reduce_sum(tf.math.lgamma(alpha),axis=1,keepdims=True)
        lnB_uni = tf.reduce_sum(tf.math.lgamma(beta),axis=1,keepdims=True) - tf.math.lgamma(S_beta)
        lnB_uni = tf.reduce_sum(tf.math.lgamma(beta),axis=1,keepdims=True) - tf.math.lgamma(S_beta)

        dg0 = tf.math.digamma(S_alpha)
        dg1 = tf.math.digamma(alpha)

        kl = tf.reduce_sum((alpha - beta)*(dg1-dg0),axis=1,keepdims=True) + lnB + lnB_uni
        return kl

    S = tf.reduce_sum(alpha, axis=1, keepdims=True)
    evidence = alpha - 1
    m = alpha / S

    A = tf.reduce_sum((y-m)**2, axis=1, keepdims=True)
    B = tf.reduce_sum(alpha*(S-alpha)/(S*S*(S+1)), axis=1, keepdims=True)

    # annealing_coef = tf.minimum(1.0,tf.cast(global_step/annealing_step,tf.float32))
    alpha_hat = y + (1-y)*alpha
    C = KL(alpha_hat)

    C = tf.reduce_mean(C, axis=1)
    return tf.reduce_mean(A + B + C)

def Sigmoid_CE(y, y_logits):
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=y_logits)
    return tf.reduce_mean(loss)
