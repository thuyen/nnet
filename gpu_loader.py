from torch.autograd import Variable
import threading
import sys
if sys.version_info[0] == 2:
    import Queue as queue
else:
    import queue

class GpuLoader(object):
    """
    https://github.com/pytorch/pytorch/blob/master/torch/utils/data/dataloader.py
    Convert a dataloader above to a gpu dataloader
    This is useful when we want to do feature extraction on a gpu,
    buffer the feature and feed it to a model on the main gpu (id=0)

    Arguments:
        model (nn.Module): feature extraction network
        loader (dataloader object)
        for_train (bool):  train or validation
        buffer_size (int): size of the prefetched buffer
        gpu_id (int): gpu that performs feature extraction

    Usage:
        train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(traindir, transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

        extractor = .... resnet for example
        train_loader = GpuLoader(extractor, train_loader)

        for input, label in train_loader:
            #train your lstm here

    """

    def __init__(self, model, loader, for_train=True, buffer_size=2, gpu_id=1):
        # Turn off BP for feature extaction module
        for param in model.parameters():
            param.requires_grad = False
        self.model = model.cuda(gpu_id)
        self.loader = loader
        self.buffer_size = buffer_size
        self.gpu_id = gpu_id
        self.volatile = not for_train

    def __iter__(self):
        def _thread(loader, buffer):
            for input, label in loader:
                input = Variable(input.cuda(self.gpu_id), volatile=self.volatile)
                output = self.model(input)
                buffer.put((output, label), block=True)
            buffer.put(None)

        buffer = queue.Queue(maxsize=self.buffer_size-1)
        thread = threading.Thread(target=_thread, args=(self.loader, buffer))
        thread.daemon = True
        thread.start()

        for input, label in iter(buffer.get, None):
            yield input.cuda(0), label.cuda(0)

    def __len__(self):
        return len(self.loader)
