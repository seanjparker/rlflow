import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp


class MDRNNBase(snt.Module):
    def __init__(self, latents, actions, hiddens, gaussians):
        super().__init__()
        self.latents = latents
        self.actions = actions
        self.hiddens = hiddens
        self.gaussians = gaussians

        self.gmm_linear = snt.Linear((2 * latents + 1) * gaussians + 2)

    def __call__(self, *inputs):
        raise NotImplementedError


class MDRNN(MDRNNBase):
    """ MDRNN model """
    def __init__(self, latents, actions, hiddens, gaussians):
        super().__init__(latents, actions, hiddens, gaussians)
        self.rnn = snt.LSTM(hiddens)

    def initial_state(self):
        return self.rnn.initial_state(1)

    def __call__(self, action, latent, prev_state):
        """ Single step
        :args actions: (batch_size, action_size) tensor
        :args latents: (batch_size, latents_size) tensor
        :args hidden: (batch_size, LSTMState)
        :returns: mu_nlat, sig_nlat, pi_nlat, r, d, next_hidden, parameters of
        the GMM prediction for the next latent, gaussian prediction of the
        reward, logit prediction of terminality and next hidden state.
            - mu_nlat: (batch_size, n_gauss, latent_size) tensor
            - sigma_nlat: (batch_size, n_gauss, latent_size) tensor
            - logpi_nlat: (batch_size, n_gauss) tensor
            - rs: (batch_size) tensor
            - ds: (batch_size) tensor
            - next_state: (LSTMState) Next state of the LSTM Cell
        """
        in_all = tf.concat([action, latent], axis=1)
        out_rnn, next_state = self.rnn(in_all, prev_state)

        out_full = self.gmm_linear(out_rnn)

        stride = self.gaussians * self.latents

        mus = out_full[:, :stride]
        mus = tf.reshape(mus, [-1, self.gaussians, self.latents])

        sigmas = out_full[:, stride:2 * stride]
        sigmas = tf.exp(tf.reshape(sigmas, [-1, self.gaussians, self.latents]))

        pi = out_full[:, 2 * stride:2 * stride + self.gaussians]
        pi = tf.nn.log_softmax(tf.reshape(pi, [-1, self.gaussians]), axis=-1)

        rs = out_full[:, -2]
        ds = out_full[:, -1]

        return mus, sigmas, pi, rs, ds, next_state


def gmm_loss(batch, mus, sigmas, log_pi, reduce=True):
    """ Computes the gmm loss.
    Compute negative log probability of a batch for the GMM model.
    Precisely, with bs1, bs2, ... the sizes of the batch
    dimensions (several batch dimension are useful when you have both a batch
    axis and a time step axis), gs the number of mixtures and fs the number of
    features.
    :args batch: (bs1, bs2, *, fs) tensor
    :args mus: (bs1, bs2, *, gs, fs) tensor
    :args sigmas: (bs1, bs2, *, gs, fs) tensor
    :args logpi: (bs1, bs2, *, gs) tensor
    :args reduce: if not reduce, the mean in the following formula is ommited
    :returns:
    loss(batch) = - mean_{i1=0..bs1, i2=0..bs2, ...} log(
        sum_{k=1..gs} pi[i1, i2, ..., k] * N(
            batch[i1, i2, ..., :] | mus[i1, i2, ..., k, :], sigmas[i1, i2, ..., k, :]))
    NOTE: The loss is not reduced along the feature dimension (i.e. it should scale ~linearily
    with fs).
    """

    batch = tf.expand_dims(batch, axis=-2)
    normal_dist = tfp.distributions.Normal(mus, sigmas)
    log_probs = log_pi + tf.reduce_sum(normal_dist.log_prob(batch), axis=-1)
    max_log_probs = tf.reduce_max(log_probs, axis=-1, keepdims=True)
    log_probs = log_probs - max_log_probs

    e_probs = tf.exp(log_probs)
    probs = tf.reduce_sum(e_probs, axis=-1)

    log_prob = tf.expand_dims(max_log_probs, axis=-1) + tf.math.log(probs)
    return -tf.reduce_mean(log_prob) if reduce else -log_prob

# if __name__ == '__main__':
#     mdrnn = MDRNN(2, 3, 8, 5)
#     mdrnn_init = mdrnn.initial_state()
#     _latents = tf.convert_to_tensor(np.random.randn(1, 2), dtype=tf.float32)
#     _mu, _sigma, _pi, _r, _d, state = mdrnn(tf.convert_to_tensor([[0.0] * 3]), _latents, mdrnn_init)
