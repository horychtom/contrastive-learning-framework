"""Module containing utility functions."""


def get_dataset_name(path_to_dataset: str):
    """
    Returns the name of the dataset given its path.

    Args:
        path_to_dataset (str): The path to the dataset.

    Returns:
        str: The name of the dataset.
    """
    return path_to_dataset.split("/")[-1].split(".")[0]
