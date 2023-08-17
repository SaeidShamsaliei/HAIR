import imp
from .cityscapes import Cityscapes
from .deepglobe import Deepglobe
from .river_segment import RiverSegment
from .river_segment2 import RiverSegment2
from .deepglobe_river import DeepGlobeRiver


NAME2DATASET = {"deepglobe": Deepglobe,
                "cityscapes": Cityscapes, "river_segment": RiverSegment, 
                "river_segment2": RiverSegment2,
                "deepglobe_river":DeepGlobeRiver,}


def get_dataset_with_name(dataset_name):
    """Get the dataset class from name

    Args:
        dataset_name (str): defined name of the dataset

    Raises:
        ValueError: when not found the dataset

    Returns:
        nn.Dataset class: class of found dataset
    """
    if dataset_name in NAME2DATASET:
        return NAME2DATASET[dataset_name]
    else:
        raise ValueError("Cannot found dataset " + dataset_name)
