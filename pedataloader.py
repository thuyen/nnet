import sys
if sys.version_info[0] == 2:
    import Queue as queue
else:
    import queue
import math
import multiprocessing as mp
import threading
import torch


class PEDataLoader(object):
    """
    A multiprocess-dataloader that paralles over elements as suppose to
    over batches (the torch built-in one)
    """

    def generate_batches(self):
        self.indices = \
                torch.randperm(self.num_samples).long() if self.shuffle else \
                torch.LongTensor(range(self.num_samples))

        for b in range(self.num_batches):
            start_index = b*self.batch_size
            end_index = (b+1)*self.batch_size if b < self.num_batches - 1 \
                    else self.num_samples
            indices = self.indices[start_index:end_index]
            batch = self.pool.map(self.dataset, indices)
            batch = self.collate_fn(batch)
            batch = self.pin_memory_fn(batch)
            yield batch

    def start(self):
        def _thread():
            for b in self.generate_batches():
                self.buffer.put(b, block=True)
            self.buffer.put(None)

        thread = threading.Thread(target=_thread)
        thread.daemon = True
        thread.start()

    def __init__(self, dataset, batch_size=1, shuffle=False,
                 num_workers=None, pin_memory=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

        self.collate_fn = torch.utils.data.dataloader.default_collate
        self.pin_memory_fn = \
                torch.utils.data.dataloader.pin_memory_batch if pin_memory else \
                lambda x: x

        self.num_samples = len(dataset)
        self.num_batches = int(math.ceil(self.num_samples / float(self.batch_size)))
        print(self.num_samples)

        self.pool = mp.Pool(num_workers)

        self.buffer = queue.Queue(maxsize=1)
        self.start()

    def __next__(self):
        batch = self.buffer.get()
        if batch is None:
            self.start()
            raise StopIteration
        return batch

    next = __next__

    def __iter__(self):
        return self

    def __len__(self):
        return self.num_batches
