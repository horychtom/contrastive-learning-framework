"""This module contains PyTorch Dataset class for loading data.""" ""
from torch.utils.data import DataLoader, Dataset


class CustomTorchDataset(Dataset):
    """PyTorch Dataset class for loading data."""

    def __init__(self, features, labels):
        """Initialize a  CustomTorchDataset object."""
        self.features = features
        self.labels = labels

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.labels)

    def __getitem__(self, index):
        """Return a sample from the dataset."""
        sample = self.features[index]
        input_ids = sample["input_ids"]
        attention_mask = sample["attention_mask"]
        labels = self.labels[index]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def get_dataloader(self, batch_size):
        """Return a DataLoader object for the dataset."""
        return DataLoader(self, batch_size=batch_size, shuffle=True)
