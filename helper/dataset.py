import os
import glob
import pyarrow
import torch


class DistributedDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset_dir, world_size, rank, batch_size, sequence_length):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.world_size = world_size
        self.rank = rank
        self.batch_size = batch_size
        self.sequence_length = sequence_length

        num_chunks = len(glob.glob(os.path.join(self.dataset_dir, "chunk.*.arrow")))
        assert num_chunks % world_size == 0, "Number of chunks must be divisible by world size."
        self.num_readers = num_chunks // world_size

        self.readers = []
        for i in range(self.num_readers):
            reader = pyarrow.ipc.open_file(pyarrow.memory_map(os.path.join(self.dataset_dir, f"chunk.{rank + i * world_size:05d}.arrow")))
            self.readers.append(reader)

        self.buffer = []
        self.global_idx = 0

    def __iter__(self):
        while True:
            reader_idx, local_idx = self.global_idx % self.num_readers, self.global_idx // self.num_readers
            sample = self.readers[reader_idx].get_batch(local_idx)['input_ids'].to_pylist()[0][:self.sequence_length + 1]
            
            self.buffer.append(sample)
            self.global_idx += 1

            while len(self.buffer) >= self.batch_size:
                input, label = [x[:-1] for x in self.buffer[:self.batch_size]], [x[1:] for x in self.buffer[:self.batch_size]]
                yield torch.tensor(input, dtype=torch.long), torch.tensor(label, dtype=torch.long)
                self.buffer = self.buffer[self.batch_size:]

    def state_dict(self):
        return {
            'buffer': self.buffer,
            'global_idx': self.global_idx,
        }

    def load_state_dict(self, state_dict):
        self.buffer = state_dict['buffer']
        self.global_idx = state_dict['global_idx']
