import torch
from tqdm import tqdm


class MMDWAE():
    def __init__(self, lamda_coeff, kernel, cost, encoder, decoder, sample_latent_prior, trainloader, optimizer, device):
        # TODO: replace this dirty hack with something normal
        vars = locals()
        self.__dict__.update(vars)
        del self.__dict__["self"]
        self.history = []


    # TODO: add num_samples
    def train(self, num_epoches): 
        for epoch in tqdm(range(num_epoches)):
            for X, Y in self.trainloader:
                batch_size = X.shape[0]
                X, Y = X.to(self.device), Y.to(self.device)
                Y = X # TODO: remove, just for testing with build-in torchvision MNIST
                Z_prior = self.sample_latent_prior(batch_size)
                Z_conditional = self.encoder(X.transpose(1,2))
                Y_pred = self.decoder(Z_conditional)
                cost_term = self.cost(Y, Y_pred)
                prior_kernel_term = self.kernel(Z_prior, Z_prior)
                conditional_kernel_term = self.kernel(Z_conditional, Z_conditional)
                cross_kernel_term = self.kernel(Z_prior, Z_conditional)
                loss = 100*cost_term / batch_size + \
                       self.lamda_coeff / batch_size / (batch_size - 1) * (prior_kernel_term + conditional_kernel_term) - \
                       2 * self.lamda_coeff / (batch_size ** 2) * cross_kernel_term
                self.history.append(loss)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()


    def sample(self, num_samples):
        Z_sampled = self.sample_latent_prior(num_samples)
        Y_sampled = self.decoder(Z_sampled)
        return Y_sampled

