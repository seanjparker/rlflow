import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp


class MDRNNBase(snt.Module):
    def __init__(self, num_latents, num_hiddens, num_gaussians):
        super().__init__()
        self.num_latents = num_latents
        self.num_hiddens = num_hiddens
        self.num_gaussians = num_gaussians

        self.gmm_linear = snt.Linear((2 * num_latents + 1) * num_gaussians + 2)

    def __call__(self, *inputs):
        raise NotImplementedError


class MDRNN(MDRNNBase):
    """ MDRNN model """
    def __init__(self, batch_size, num_latents, num_actions, num_hiddens, num_gaussians):
        super().__init__(num_latents, num_hiddens, num_gaussians)
        self.rnn = snt.LSTM(num_hiddens)
        self.batch_size = batch_size

    def initial_state(self):
        return self.rnn.initial_state(self.batch_size)

    def __call__(self, latents, prev_state):
        """ Single step
        :args latents: (batch_size, latents_size) tensor
        :args prev_state: (batch_size, LSTMState) tensor
        :returns: mu_nlat, sig_nlat, pi_nlat, r, d, next_hidden, parameters of
        the GMM prediction for the next latent, gaussian prediction of the
        reward, logit prediction of terminality and next hidden state.
            - mu_nlat: (batch_size, n_gauss, latent_size) tensor
            - sigma_nlat: (batch_size, n_gauss, latent_size) tensor
            - logpi_nlat: (batch_size, n_gauss) tensor
            - rewards: (batch_size) tensor
            - dones: (batch_size) tensor
            - next_state: (LSTMState) Next state of the LSTM Cell
        """
        out_rnn, next_state = self.rnn(latents, prev_state)

        out_full = self.gmm_linear(out_rnn)

        stride = self.num_gaussians * self.num_latents

        mus = out_full[:, :stride]
        mus = tf.reshape(mus, [-1, self.num_gaussians, self.num_latents])

        sigmas = out_full[:, stride:2 * stride]
        sigmas = tf.exp(tf.reshape(sigmas, [-1, self.num_gaussians, self.num_latents]))

        pi = out_full[:, 2 * stride:2 * stride + self.num_gaussians]
        pi = tf.nn.log_softmax(tf.reshape(pi, [-1, self.num_gaussians]), axis=-1)

        rewards = out_full[:, -2]
        dones = out_full[:, -1]

        return (mus, sigmas, pi, rewards, dones), next_state


def gmm_loss(next_latent_obs, mus, sigmas, log_pi, reduce=True):
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

    next_latent_obs = tf.expand_dims(next_latent_obs, axis=-2)
    normal_dist = tfp.distributions.Normal(mus, sigmas)
    log_probs = log_pi + tf.reduce_sum(normal_dist.log_prob(next_latent_obs), axis=-1)
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
