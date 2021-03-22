import sonnet as snt
import tensorflow as tf


class Controller(snt.Module):
    def __init__(self, num_actions, name='sonnet_Controller'):
        super(Controller, self).__init__(name=name)
        self.network = snt.Linear(num_actions)

    def __call__(self, g_latents, m_latents):
        """
        Forward pass through the controller using the output of the graph embedding and MDRNN modules
        Args:
            g_latents: latent tensor from the GNN
            m_latents: latent tensor from the MDRNN

        Returns: logits based on the latent variables from the graph and memory modules
        """
        cat_in = tf.concat([g_latents, m_latents], dim=1)
        return self.network(cat_in)
