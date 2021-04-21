import numpy as np
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
    def __init__(self, batch_size, num_latents, num_actions, num_hiddens, num_gaussians, training=False):
        super().__init__(num_latents, num_hiddens, num_gaussians)
        # Use an snt.UnrolledLSTM as it is an order of magnitude faster then snt.dynamic_unroll with a LSTM core
        # + need to support different sequence length and batch sizes
        self.unrolled_lstm = snt.UnrolledLSTM(num_hiddens)
        self.batch_size = batch_size
        self.seq_len = 256  # TODO: fix hardcoded lstm state length
        self.last_state = self._initial_state()
        self.training = training

    def _initial_state(self):
        return self.unrolled_lstm.initial_state(self.batch_size)

    def __call__(self, xfer_actions, loc_actions, latents):
        """ Multiple steps
        :args xfer_actions: (batch_size, seq_len, actions_size) tensor
        :args loc_actions: (batch_size, seq_len, actions_size) tensor
        :args latents: (batch_size, seq_len, latents_size) tensor
        :args prev_state: (batch_size, LSTMState) tensor
        :returns: mu_nlat, sig_nlat, pi_nlat, r, d, next_hidden, parameters of
        the GMM prediction for the next latent, gaussian prediction of the
        reward, logit prediction of terminality and next hidden state.
            - mu_nlat: (seq_len, batch_size, n_gauss, latent_size) tensor
            - sigma_nlat: (seq_len, batch_size, n_gauss, latent_size) tensor
            - logpi_nlat: (seq_len, batch_size, n_gauss) tensor
            - rewards: (seq_len, batch_size) tensor
            - dones: (seq_len, batch_size) tensor
            - next_state: (LSTMState) Next state of the LSTM Cell
        """

        def pad(t, max_in_dims):
            s = tf.shape(t)
            paddings = [[0, m - s[i]] for i, m in enumerate(max_in_dims)]
            return tf.pad(t, paddings)

        def convert_actions(tensor):
            # Try to convert to a RaggedTensor to support both batched and single-steps in the world model
            tensor = tf.ragged.constant(tensor, dtype=tf.float32)
            if isinstance(tensor, tf.RaggedTensor):
                tensor = tensor.to_tensor(shape=[None, self.seq_len])
            elif isinstance(tensor, tf.Tensor):
                tensor = pad(tf.expand_dims(tensor, axis=-1), [1, self.seq_len])
            return tf.expand_dims(tensor, axis=-1)  # Shape (batch_size, seq_length, 1)

        def convert_latents(tensor):
            if isinstance(tensor, list):
                tensor = tf.ragged.constant(
                    tensor, dtype=tf.float32).to_tensor(shape=[None, self.seq_len, self.num_latents])
            elif isinstance(tensor, tf.Tensor):
                tensor = pad(tensor, [1, self.seq_len, self.num_latents])
            return tensor

        padded_xfer_actions = convert_actions(xfer_actions)
        padded_loc_actions = convert_actions(loc_actions)
        padded_latents = convert_latents(latents)

        padded_latents = tf.transpose(padded_latents, [1, 0, 2])
        padded_xfer_actions = tf.transpose(padded_xfer_actions, [1, 0, 2])
        padded_loc_actions = tf.transpose(padded_loc_actions, [1, 0, 2])

        input_sequence = tf.concat([padded_xfer_actions, padded_loc_actions, padded_latents], axis=-1)

        # input_sequence -- Tensor shape (seq_len, batch_size, ...)
        out_rnn, next_state = self.unrolled_lstm(input_sequence, self.last_state)
        self.last_state = next_state

        out_full = self.gmm_linear(out_rnn)

        stride = self.num_gaussians * self.num_latents

        mus = out_full[:, :, :stride]
        mus = tf.reshape(mus, [self.seq_len, self.batch_size, self.num_gaussians, self.num_latents])

        sigmas = out_full[:, :, stride:2 * stride]
        sigmas = tf.exp(tf.reshape(sigmas, [self.seq_len, self.batch_size, self.num_gaussians, self.num_latents]))

        pi = out_full[:, :, 2 * stride:2 * stride + self.num_gaussians]
        pi = tf.nn.log_softmax(tf.reshape(pi, [self.seq_len, self.batch_size, self.num_gaussians]), axis=-1)

        rs = out_full[:, :, -2]
        ds = out_full[:, :, -1]

        return (mus, sigmas, pi, rs, ds), next_state


# TODO: infer num_latents, seq_len, num_gaussians from tensor shapes
def gmm_loss(next_latents, mus, sigmas, log_pi, reduce=True, num_latents=1600, seq_len=256, num_gaussians=8):
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

    next_latents = tf.ragged.constant(next_latents, dtype=tf.float32).to_tensor(
        shape=[None, seq_len, num_latents])
    next_latents = tf.expand_dims(
        tf.transpose(next_latents, [1, 0, 2]), axis=-2)
    normal_dist = tfp.distributions.Normal(mus, sigmas)
    logs = normal_dist.log_prob(next_latents)
    log_probs = log_pi + tf.reduce_sum(logs, axis=-1)
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
