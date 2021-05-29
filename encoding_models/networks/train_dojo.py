import torch
from .early_stopping import EarlyStopping

class TrainDojo(object):
    """ Learning Loop for basic regression/classification setups """
    def __init__(self, network, optimizer, criterion,
                 train_loader, test_loader=None,
                 log_batch_interval=None, scheduler=None,
                 patience=7, device=torch.device('cpu')):

        self.network = network              # Network to train
        self.criterion = criterion          # Loss criterion to minimize
        self.device = device                # Device for tensors
        self.optimizer = optimizer          # PyTorch optimizer for SGD
        self.scheduler = scheduler          # Learning rate scheduler
        self.patience = patience            # Early stopping patience
        if self.patience is not None:
            self.early_stopping = EarlyStopping(patience=patience, verbose=True)

        self.train_loader = train_loader
        self.test_loader = test_loader

        self.log_batch_interval = log_batch_interval
        self.batch_processed = 0

    def train(self, num_epochs):
        """ Loop over epochs in training & test on all hold-out batches """
        # Get Initial Performance after Network Initialization
        train_performance = self.get_network_performance(test=False)
        test_performance = self.get_network_performance(test=True)
        if self.patience is not None:
            self.early_stopping(test_performance, self.network)

        for epoch_id in range(1, num_epochs+1):
            # Train the network for a single epoch
            self.train_for_epoch(epoch_id)
            if self.patience is not None:
                if self.early_stopping.early_stop:
                    return - self.early_stopping.best_score

            # Update the learning rate using the scheduler (if desired)
            if self.scheduler is not None:
                self.scheduler.step()
        if self.patience is not None:
            return - self.early_stopping.best_score

    def train_for_epoch(self, epoch_id=0):
        """ Perform one epoch of training with train data loader """
        self.network.train()
        # Loop over batches in the training dataset
        for batch_idx, (data, target) in enumerate(self.train_loader):
            # Put data on the device
            data, target = data.to(self.device), target.to(self.device)
            # Clear gradients & perform forward as well as backward pass
            self.optimizer.zero_grad()
            output = self.network(data.float())
            loss = self.criterion(output, target.float())
            loss.backward()
            self.optimizer.step()

            # Update the batches processed counter
            self.batch_processed += 1

            # Log the performance of the network
            if self.batch_processed % self.log_batch_interval == 0:
                # Get Current Performance after Single Epoch of Training
                train_performance = self.get_network_performance(test=False)
                test_performance = self.get_network_performance(test=True)
                # early_stopping needs the validation loss to check if it has decresed,
                # and if it has, it will make a checkpoint of the current model
                self.early_stopping(test_performance, self.network)
                if self.early_stopping.early_stop:
                    break


    def get_network_performance(self, test=False):
        """ Get the performance of the network """
        loader = self.test_loader if test else self.train_loader
        self.network.eval()
        loss, b_counter = 0, 0
        with torch.no_grad():
            # Loop over batches and get average batch accuracy/loss
            for data, target in loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.network(data.float())
                target = target.float()
                loss += self.criterion(output, target).sum().item()
                b_counter += 1

        avg_loss = loss/b_counter
        return avg_loss
