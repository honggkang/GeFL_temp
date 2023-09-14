import torch
import torch.nn as nn

# temporary roundabout to evaluate sensitivity of the generator
GENERATORCONFIGS = {
    # hidden_dimension, latent_dimension, input_channel, n_class, noise_dim
    'cifar': (512, 32, 3, 10, 64),
    'celeb': (128, 32, 3, 2, 32),
    'mnist': (256, 32, 1, 10, 32),
    'mnist-cnn0': (256, 32, 1, 10, 64),
    'mnist-cnn1': (128, 32, 1, 10, 32),
    'mnist-cnn2': (64, 32, 1, 10, 32),
    'mnist-cnn3': (64, 32, 1, 10, 16),
    'emnist': (256, 32, 1, 26, 32),
    'emnist-cnn0': (256, 32, 1, 26, 64),
    'emnist-cnn1': (128, 32, 1, 26, 32),
    'emnist-cnn2': (128, 32, 1, 26, 16),
    'emnist-cnn3': (64, 32, 1, 26, 32),
}

class SimGenerator(nn.Module):
    def __init__(self, args, embedding=False) -> None:
        super(SimGenerator, self).__init__()
        self.embedding = embedding
        self.hidden_dim, self.latent_dim, self.input_channel, self.n_class, self.noise_dim = GENERATORCONFIGS[args.dataset]
        input_dim = self.noise_dim *2 if self.embedding else self.noise_dim + self.n_class
        self.fc_configs = [input_dim, self.hidden_dim]
        
        if self.embedding:
            self.embedding_layer = nn.Embedding(self.n_class, self.noise_dim)
            
        self.fc_layers = nn.ModuleList()
        for i in range(len(self.fc_configs)-1):
            input_dim, out_dim = self.fc_configs[i], self.fc_configs[i+1]
            print("Build layer {} X {}".format(input_dim, out_dim))
            fc = nn.Linear(input_dim, out_dim)
            bn = nn.BatchNorm1d(out_dim)
            act = nn.ReLU()
            self.fc_layers += [fc, bn, act]
        self.representation_layer = nn.Linear(self.fc_configs[-1], self.latent_dim)


    def forward(self, labels, latent_layer_idx=-1, verbose=True):
        """
        G(Z|y) or G(X|y):
        Generate either latent representation( latent_layer_idx < 0) or raw image (latent_layer_idx=0) conditional on labels.
        :param labels:
        :param latent_layer_idx:
            if -1, generate latent representation of the last layer,
            -2 for the 2nd to last layer, 0 for raw images.
        :param verbose: also return the sampled Gaussian noise if verbose = True
        :return: a dictionary of output information.
        """
        result = {}
        batch_size = labels.shape[0]
        eps = torch.rand((batch_size, self.noise_dim)) # sampling from Gaussian
        if verbose:
            result['eps'] = eps
        if self.embedding: # embedded dense vector
            y_input = self.embedding_layer(labels)
        else: # one-hot (sparse) vector
            y_input = torch.FloatTensor(batch_size, self.n_class)
            y_input.zero_()
            #labels = labels.view
            y_input.scatter_(1, labels.view(-1,1), 1)
        z = torch.cat((eps, y_input), dim=1)
        ### FC layers
        for layer in self.fc_layers:
            z = layer(z)
        z = self.representation_layer(z)
        result['output'] = z
        return result
    
# gen_output=self.generative_model(y, latent_layer_idx=self.latent_layer_idx)['output']
