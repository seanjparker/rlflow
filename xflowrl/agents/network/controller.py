import sonnet as snt


class Controller(snt.Module):
    def __init__(self, num_actions, name='sonnet_Controller'):
        super(Controller, self).__init__(name=name)
        self.network = snt.Linear(num_actions)

    def __call__(self, g_latents):
        """
        Forward pass through the controller using the output of the graph embedding produced by the world model
        Args:
            g_latents: latent tensor

        Returns: logits based on the latent variables from the graph and memory modules
        """
        return self.network(g_latents)
