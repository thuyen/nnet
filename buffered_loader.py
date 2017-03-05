import sys
if sys.version_info[0] == 2:
    import Queue as queue
else:
    import queue
import threading

import torch.utils.data as data

class BDataLoader(object):
    """
    Buffered dataloader
    Similar to the torch DataLoader class but initialize the iterator in advance
    This will start preparing the first batch at initialization
    """

    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None,
                 num_workers=0, pin_memory=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.collate_fn = data.dataloader.default_collate
        self.pin_memory = pin_memory

        if sampler is not None:
            self.sampler = sampler
        elif shuffle:
            self.sampler = data.sampler.RandomSampler(dataset)
        elif not shuffle:
            self.sampler = data.sampler.SequentialSampler(dataset)

        def _thread():
            while True:
                self.buffer.put(data.dataloader.DataLoaderIter(self), block=True)

        self.buffer = queue.Queue(maxsize=1)
        thread = threading.Thread(target=_thread)
        thread.daemon = True
        thread.start()

    def __iter__(self):
        return self.buffer.get()

    def __len__(self):
        return int(math.ceil(len(self.sampler) / float(self.batch_size)))
