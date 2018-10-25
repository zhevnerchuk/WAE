import torch


class GANWAE():
    def __init__(self, lamda_coeff, divergence, cost, encoder, decoder, sample_latent_prior, trainloader, optimizer, device):
        # TODO: replace this dirty hack with something normal
        vars = locals()
        self.__dict__.update(vars)
        del self.__dict__["self"]
        self.history = []


    # TODO: add num_samples
    def train(self, num_epoches): 
        for epoch in range(num_epoches):
            for X, Y in self.trainloader:
                batch_size = X.shape[0]
                X, Y = X.to(self.device), Y.to(self.device)
                Y = X # TODO: remove, just for testing with build-in torchvision MNIST
                Z_prior = self.sample_latent_prior(batch_size)
                Z_conditional = self.encoder(X)
                Y_pred = self.decoder(Z_conditional)
                cost_term = self.cost(Y, Y_pred)
                loss = cost_term / batch_size + self.lamda_coeff * self.divergence(Z_prior, Z_conditional)
                self.history.append(loss)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()


    def sample(self, num_samples):
        Z_sampled = self.sample_latent_prior(num_samples)
        Y_sampled = self.decoder(Z_sampled)
        return Y_sampled

