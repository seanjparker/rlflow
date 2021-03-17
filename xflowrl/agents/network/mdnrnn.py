import sonnet as snt
import tensorflow as tf


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
        self.prev_state = 0

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
        in_al = tf.concat([action, latent], axis=1)

        out_rnn, next_state = self.rnn(in_al, prev_state)

        out_full = self.gmm_linear(out_rnn)

        stride = self.gaussians * self.latents

        mus = out_full[:, :stride]
        mus = mus.reshape(-1, self.gaussians, self.latents)

        sigmas = out_full[:, stride:2 * stride]
        sigmas = sigmas.reshape(-1, self.gaussians, self.latents)
        sigmas = tf.exp(sigmas)

        pi = out_full[:, 2 * stride:2 * stride + self.gaussians]
        pi = pi.reshape(-1, self.gaussians)
        log_pi = tf.nn.log_softmax(pi, axis=-1)

        r = out_full[:, -2]

        d = out_full[:, -1]

        return mus, sigmas, log_pi, r, d, next_state
