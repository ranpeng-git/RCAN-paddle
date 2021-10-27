import sys
import threading
import queue
import random
import collections

import paddle
import multiprocessing

from paddle.io import DataLoader 
from paddle.fluid.dataloader.dataloader_iter import _DataLoaderIterBase ,default_collate_fn
from paddle.fluid.multiprocess_utils import _set_SIGCHLD_handler

# from torch.utils.data.dataloader import ExceptionWrapper
# from torch.utils.data.dataloader import _set_SIGCHLD_handler

if sys.version_info[0] == 2:
    import Queue as queue
else:
    import queue

def _ms_loop(dataset, index_queue, data_queue, collate_fn, scale, seed, init_fn, worker_id):
    # paddle.set_num_threads(1)
    paddle.seed(seed)
    while True:
        r = index_queue.get()
        if r is None:
            break
        idx, batch_indices = r
        try:
            idx_scale = 0
            if len(scale) > 1 and dataset.train:
                idx_scale = random.randrange(0, len(scale))
                dataset.set_scale(idx_scale)

            samples = collate_fn([dataset[i] for i in batch_indices])
            samples.append(idx_scale)

        except Exception:
            data_queue.put((idx, ExceptionWrapper(sys.exc_info())))
        else:
            data_queue.put((idx, samples))

class _MSDataLoaderIter(_DataLoaderIterBase):
    def __init__(self, loader):
        self.dataset = loader.dataset
        self.scale = loader.scale
        self.collate_fn = loader.collate_fn
        self.batch_sampler = loader.batch_sampler
        self.num_workers = loader.num_workers
        self.timeout = loader.timeout
        self.done_event = threading.Event()

        self.sample_iter = iter(self.batch_sampler)

        if self.num_workers > 0:
            self.worker_init_fn = loader.worker_init_fn
            self.index_queues = [
                multiprocessing.Queue() for _ in range(self.num_workers)
            ]
            self.worker_queue_idx = 0
            self.worker_result_queue = multiprocessing.SimpleQueue()
            self.batches_outstanding = 0
            self.worker_pids_set = False
            self.shutdown = False
            self.send_idx = 0
            self.rcvd_idx = 0
            self.reorder_dict = {}

            base_seed = torch.LongTensor(1).random_()[0]
            self.workers = [
                multiprocessing.Process(
                    target=_ms_loop,
                    args=(
                        self.dataset,
                        self.index_queues[i],
                        self.worker_result_queue,
                        self.collate_fn,
                        self.scale,
                        base_seed + i,
                        self.worker_init_fn,
                        i
                    )
                )
                for i in range(self.num_workers)]

            if self.pin_memory or self.timeout > 0:
                self.data_queue = queue.Queue()
                if self.pin_memory:
                    maybe_device_id = torch.cuda.current_device()
                else:
                    # do not initialize cuda context if not necessary
                    maybe_device_id = None
                self.worker_manager_thread = threading.Thread(
                    target=_worker_manager_loop,
                    args=(self.worker_result_queue, self.data_queue, self.done_event, self.pin_memory,
                          maybe_device_id))
                self.worker_manager_thread.daemon = True
                self.worker_manager_thread.start()
            else:
                self.data_queue = self.worker_result_queue

            for w in self.workers:
                w.daemon = True  # ensure that the worker exits on process exit
                w.start()

            _update_worker_pids(id(self), tuple(w.pid for w in self.workers))
            _set_SIGCHLD_handler()
            self.worker_pids_set = True

            # prime the prefetch loop
            for _ in range(2 * self.num_workers):
                self._put_indices()

# class MSDataLoader(DataLoader):
#     def __init__(
#         self, args, dataset, batch_size=1, shuffle=False,
#         sampler=None, batch_sampler=None,
#         collate_fn=default_collate, pin_memory=False, drop_last=False,
#         timeout=0, worker_init_fn=None):

#         super(MSDataLoader, self).__init__(
#             dataset, batch_size=batch_size, shuffle=shuffle,
#             sampler=sampler, batch_sampler=batch_sampler,
#             num_workers=args.n_threads, collate_fn=collate_fn,
#             pin_memory=pin_memory, drop_last=drop_last,
#             timeout=timeout, worker_init_fn=worker_init_fn)

#         self.scale = args.scale

#     def __iter__(self):
#         return _MSDataLoaderIter(self)


class MSDataLoader(DataLoader):
    def __init__(
        self, args, dataset, feed_list=None, places=None, return_list=False, batch_sampler=None,
                 batch_size=1, shuffle=True, drop_last=False, collate_fn=None, num_workers=0, 
                 use_buffer_reader=True, use_shared_memory=True, timeout=0, worker_init_fn=None):
        super(MSDataLoader, self).__init__(
             dataset, feed_list=feed_list, places=places, return_list=return_list, batch_sampler= batch_sampler,
                 batch_size=args.batch_size, shuffle=shuffle, drop_last=drop_last, collate_fn=collate_fn, num_workers=args.n_threads, 
                 use_buffer_reader=use_buffer_reader, use_shared_memory=use_shared_memory, timeout=timeout, worker_init_fn=worker_init_fn)

        self.scale = args.scale

    # def __iter__(self):
    #     return _MSDataLoaderIter(self)
