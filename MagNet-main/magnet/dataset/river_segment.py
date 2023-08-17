from .base import BaseDataset


class RiverSegment(BaseDataset):
    """Deepglobe dataset generator"""

    def __init__(self, opt):
        super().__init__(opt)

        self.label2color = {
            0: [0, 0, 0],    # unknown
            1: [255, 255, 0],  # gravel
            2: [0, 128, 0],  # vegetation
            3: [128, 128, 0],  # farmland
            4: [128, 0, 128],  # human_construction
            5: [0, 0, 255],  # water
        }
