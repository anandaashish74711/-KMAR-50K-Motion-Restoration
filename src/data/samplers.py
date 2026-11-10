import random
from torch.utils.data import Sampler


class VolumeShuffleSampler(Sampler):
    def __init__(self, dataset, seed=42):
        self.dataset = dataset
        self.seed = seed
        self.epoch = 0

        self.groups = {}
        for idx, (artifact_path, _, _) in enumerate(dataset.samples):
            self.groups.setdefault(artifact_path, []).append(idx)

        self.volumes = list(self.groups)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)

        volumes = self.volumes.copy()
        rng.shuffle(volumes)

        for volume in volumes:
            indices = self.groups[volume].copy()
            rng.shuffle(indices)
            yield from indices

    def __len__(self):
        return len(self.dataset)
