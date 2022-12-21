import torch
from torch.utils.data import Dataset

class BasicDataset(Dataset):
    def __init__(self, inputs, labels):
        self.labels = labels
        self.inputs = inputs

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        # Load data and get label
        X = self.inputs[index]
        Y = self.labels[index]
        return X, Y

class EncDecDataset(Dataset):
    def __init__(self, inputs, labels):
        self.labels = labels
        self.inputs = inputs

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        # Load data and get label
        X = self.inputs[index]
        start_token = X[-1]
        start_token = start_token[None, :]

        y_label = self.labels[index]
        y_inputs = torch.cat((start_token, y_label[:-1, :]))
        return X, (y_inputs, y_label) # X is input for encoder, y_inputs is input for decoder

# Modified the start token
class EncDecDataset2(Dataset):
    def __init__(self, inputs, labels):
        self.labels = labels
        self.inputs = inputs

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        X = self.inputs[index]
        Y = self.labels[index]

        # Use 40-50 as start token
        # Calculate loss on 50-100
        encoder_labels = torch.concat((X[1:], Y[None, 0]), dim=0)
        start_token = X[40:]
        return X, (encoder_labels, start_token, Y)

# Monitor encoder training
class EncDecDataset3(Dataset):
    def __init__(self, inputs, labels):
        self.labels = labels
        self.inputs = inputs

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        X = self.inputs[index]
        Y = self.labels[index]

        # Add a label for encoder output to monitor encoder training
        encoder_labels = torch.concat((X[1:], Y[None, 0]), dim=0)
        decoder_inputs = torch.concat((X[None, -1], Y[:-1]), dim=0)

        return X, (encoder_labels, decoder_inputs, Y)