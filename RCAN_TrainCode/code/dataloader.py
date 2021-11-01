import sys
import threading
import queue
import random
import collections
import itertools
import logging

import paddle
import multiprocessing

from paddle.io import DataLoader 
from paddle.fluid.dataloader.dataloader_iter import _DataLoaderIterBase ,default_collate_fn
from paddle.fluid.multiprocess_utils import _set_SIGCHLD_handler


import os
import six
import sys
import paddle
import numpy as np
import traceback
from collections import namedtuple
from  paddle.fluid import core
from paddle.fluid.dataloader.fetcher import _IterableDatasetFetcher, _MapDatasetFetcher
from paddle.fluid.multiprocess_utils import _cleanup_mmap, CleanupFuncRegistrar, MP_STATUS_CHECK_INTERVAL
from paddle.fluid.framework import in_dygraph_mode
from paddle.fluid.dataloader.flat import _flatten_batch , _restore_batch
from paddle.fluid.dataloader.worker import _generate_states , WorkerInfo , _DatasetKind ,_WorkerException ,ParentWatchDog ,_IterableDatasetStopIteration
from paddle.fluid.dataloader.dataloader_iter import _DataLoaderIterMultiProcess ,_DataLoaderIterSingleProcess

from paddle.fluid.framework import _set_expected_place, _current_expected_place

if six.PY2:
    import Queue as queue
else:
    import queue

def _worker_loop(dataset, dataset_kind, indices_queue, out_queue, done_event,
                 auto_collate_batch, collate_fn, init_fn, worker_id,
                 num_workers, use_shared_memory ,scale):
    try:
        # NOTE: [ mmap files clear ] When the child process exits unexpectedly,
        # some shared memory objects may have been applied for but have not yet
        # been put into the inter-process Queue. This part of the object needs
        # to be cleaned up when the process ends.
        CleanupFuncRegistrar.register(_cleanup_mmap)

        # set signal handler
        core._set_process_signal_handler()

        # set different numpy seed for each worker
        try:
            import numpy as np
            import time
        except ImportError:
            pass
        else:
            np.random.seed(_generate_states(int(time.time()), worker_id))

        global _worker_info
        _worker_info = WorkerInfo(
            id=worker_id, num_workers=num_workers, dataset=dataset)

        init_exception = None
        try:
            if init_fn is not None:
                init_fn(worker_id)
            fetcher = _DatasetKind.create_fetcher(
                dataset_kind, dataset, auto_collate_batch, collate_fn, True)
        except:
            init_exception = _WorkerException(worker_id)

        iterator_drained = False
        parent_watch_dog = ParentWatchDog()

        while parent_watch_dog.is_alive():
            try:
                data = indices_queue.get(MP_STATUS_CHECK_INTERVAL)
            except queue.Empty:
                continue

            # None as poison piil, so worker event should be set
            if data is None:
                assert done_event.is_set() or iterator_drained, \
                        "get None when worker done_event set"
                break
            # If worker done event is set but get still get data in
            # indices_queue, remaining data should be get and skipped.
            if done_event.is_set() or iterator_drained:
                continue

            idx, indices = data
            try:
                if init_exception is not None:
                    batch = init_exception
                    init_exception = None
                else:
                    # NOTE: GPU tensor operation is not supported in sub-process
                    #       but default device is GPU in paddle-gpu version, which
                    #       may copy CPU tensor to GPU even if users want to use
                    #       CPU tensor operation, so we add CPUPlace guard here
                    #       to make sure tensor will be operated only on CPU
                    with paddle.fluid.dygraph.guard(place=paddle.CPUPlace()):
                        batch = fetcher.fetch(indices)
            except Exception as e:
                if isinstance(
                        e, StopIteration) and dataset_kind == _DatasetKind.ITER:
                    out_queue.put(_IterableDatasetStopIteration(worker_id))
                    iterator_drained = True
                else:
                    out_queue.put((idx, _WorkerException(worker_id), None))
            else:
                if isinstance(batch, _WorkerException):
                    out_queue.put((idx, batch, None))
                batch, structure = _flatten_batch(batch)

                idx_scale = 0
                if len(scale) > 1 and dataset.train:
                    idx_scale = random.randrange(0, len(scale))
                    dataset.set_scale(idx_scale)
                structure.append(idx_scale)

                if use_shared_memory:
                    tensor_list = [
                        core._array_to_share_memory_tensor(b)
                        if isinstance(b, np.ndarray) else b._share_memory()
                        for b in batch
                    ]
                    out_queue.put((idx, tensor_list, structure))
                    core._remove_tensor_list_mmap_fds(tensor_list)
                else:
                    out_queue.put((idx, batch, structure))
    except KeyboardInterrupt:
        # NOTE: Main process will raise KeyboardInterrupt anyways, ignore it in child process
        pass
    except:
        six.reraise(*sys.exc_info())
    finally:
        if use_shared_memory:
            _cleanup_mmap()

class _MSDataLoaderIter(_DataLoaderIterBase):
    def __init__(self, loader):
        super(_MSDataLoaderIter, self).__init__(loader)
        assert self._num_workers > 0,  "Multi-process DataLoader " \
                    "invalid num_workers({})".format(self._num_workers)

        # subprocess wrokers' result queue
        self._data_queue = None

        # data get from _data_queue will be reordered by _rcvd_idx
        # for data order keeping, data index not equal _rcvd_idx 
        # will be cached in _task_infos
        self._send_idx = 0
        self._rcvd_idx = 0
        self._batches_outstanding = 0
        self._task_infos = {}
        self._structure_infos = []

        self.scale = loader.scale

        # indices outstand as _outstanding_capacity at first, and
        # blocking_queue capacity is also _outstanding_capacity.
        # _outstanding_capacity here to make sure each indices_queue
        # has at least 2 indices, and outstanding batch cached
        # output data for at least 2 iterations(Note that len(_places)
        # batches will be composed as an iteration output)
        self._outstanding_capacity = 2 * max(self._num_workers,
                                             len(self._places))

        # see _try_put_indices
        self._thread_lock = threading.Lock()

        # init workers and indices queues and put 2 indices in each indices queue
        self._init_workers()
        for _ in range(self._outstanding_capacity):
            self._try_put_indices()
        
        self._init_thread()
        self._shutdown = False

    def _init_workers(self):
        # multiprocess worker and indice queue list initial as empty
        self._workers = []
        self._worker_status = []
        self._indices_queues = []
        self._workers_idx_cycle = itertools.cycle(range(self._num_workers))

        # create data_queue for workers
        self._data_queue = multiprocessing.Queue()

        # event for workers and thread, thread event is only need 
        # in multi-processing mode
        self._workers_done_event = multiprocessing.Event()
        self._thread_done_event = threading.Event()

        for i in range(self._num_workers):
            indices_queue = multiprocessing.Queue()
            self._indices_queues.append(indices_queue)
            worker = multiprocessing.Process(
                target=_worker_loop,
                args=(self._dataset, self._dataset_kind, indices_queue,
                      self._data_queue, self._workers_done_event,
                      self._auto_collate_batch, self._collate_fn,
                      self._worker_init_fn, i, self._num_workers,
                      self._use_shared_memory , self.scale))
            worker.daemon = True
            worker.start()
            self._workers.append(worker)
            self._worker_status.append(True)

        core._set_process_pids(id(self), tuple(w.pid for w in self._workers))
        _set_SIGCHLD_handler()
    def _clear_and_remove_data_queue(self):
        if self._data_queue is not None:
            while True:
                try:
                    self._data_queue.get_nowait()
                except:
                    self._data_queue.cancel_join_thread()
                    self._data_queue.close()
                    break

    def _init_thread(self):
        self._var_names = [v.name for v in self._feed_list]
        self._shapes = [v.shape for v in self._feed_list]
        # print(v.shape)
        self._dtypes = [v.dtype for v in self._feed_list]
        self._need_check_feed = [
            v.desc.need_check_feed() for v in self._feed_list
        ]
        # if only 1 place, do not need to keep order
        self._blocking_queue = core.init_lod_tensor_blocking_queue(
            core.Variable(), self._outstanding_capacity, len(self._places) > 1)
        self._reader = core.create_py_reader(
            self._blocking_queue, self._var_names, self._shapes, self._dtypes,
            self._need_check_feed, self._places, self._use_buffer_reader, True,
            self._pin_memory)

        self._thread_done_event = threading.Event()
        self._thread = threading.Thread(
            target=self._thread_loop, args=(_current_expected_place(), ))
        self._thread.daemon = True
        self._thread.start()
        return self._shapes

    def _shutdown_worker(self, worker_id):
        if self._worker_status[worker_id]:
            self._indices_queues[worker_id].put(None)
            self._worker_status[worker_id] = False

    def _try_shutdown_all(self, timeout=None):
        if not self._shutdown:
            try:
                self._exit_thread_expectedly()
                self._clear_and_remove_data_queue()

                # set _workers_done_event should be set before put None
                # to indices_queue, workers wll exit on reading None from
                # indices_queue
                self._workers_done_event.set()
                for i in range(self._num_workers):
                    self._shutdown_worker(i)

                if not self._shutdown:
                    for w in self._workers:
                        w.join(timeout)
                    for q in self._indices_queues:
                        q.cancel_join_thread()
                        q.close()
            finally:
                core._erase_process_pids(id(self))
                self._shutdown = True

    def _exit_thread_expectedly(self):
        self._thread_done_event.set()
        self._blocking_queue.close()

    def _exit_thread_unexpectedly(self):
        self._thread_done_event.set()
        self._blocking_queue.kill()
        logging.error("DataLoader reader thread raised an exception!")     

    def _thread_loop(self, legacy_expected_place):
        #NOTE(zhiqiu): Set the expected place for new thread as the same as father thread,
        # and it will call platform::SetDeviceId() in c++ internally.
        # If we do not set cudaDeviceId in new thread, the default cudaDeviceId will be 0,
        # Which may cost hundreds of MB of GPU memory on CUDAPlace(0) if calling some cuda 
        # APIs in this thread.
        _set_expected_place(legacy_expected_place)

        while not self._thread_done_event.is_set():
            batch = self._get_data()
            if not self._thread_done_event.is_set():
                if batch is None:
                    self._exit_thread_expectedly()
                else:
                    try:
                        # pack as LoDTensorArray
                        array = core.LoDTensorArray()
                        if self._use_shared_memory:
                            for tensor in batch:
                                array.append(tensor)
                        else:
                            # LoDTensor not in shared memory is not
                            # serializable, cannot be create in workers
                            for slot in batch:
                                if isinstance(slot, paddle.Tensor):
                                    slot = slot.value().get_tensor()
                                elif not isinstance(slot, core.LoDTensor):
                                    tmp = core.LoDTensor()
                                    tmp.set(slot, core.CPUPlace())
                                    slot = tmp
                                array.append(slot)

                        if not self._blocking_queue.push(array):
                            self._blocking_queue.close()
                    except:
                        self._exit_thread_unexpectedly()
                        six.reraise(*sys.exc_info())
                    finally:
                        self._rcvd_idx += 1

    def _get_data(self):
        while not self._thread_done_event.is_set():
            # For IterableDataset, batch indices is generated infinitely
            # for each worker to raise StopIteration, but a StopIteration
            # raising process will discard a batch indices which is count
            # in _send_idx but will not increase _rcvd_idx, so we check 
            # whether the worker is still alive here to skip the discarded
            # batch indices and increase _rcvd_idx
            if self._dataset_kind == _DatasetKind.ITER:
                while self._rcvd_idx < self._send_idx:
                    sys.stdout.flush()
                    info = self._task_infos[self._rcvd_idx]
                    if len(info) == 3 or self._worker_status[info[0]]:
                        break
                    del self._task_infos[self._rcvd_idx]
                    self._rcvd_idx += 1
                    self._batches_outstanding -= 1
                else:
                    # NOTE: _rcvd_idx and _send_idx only record batches among
                    #       workers, if batches among workers drained, there
                    #       may also be data in blocking queue
                    if self._batches_outstanding < len(self._places):
                        return None
                    continue

            if self._rcvd_idx in self._task_infos and \
                    len(self._task_infos[self._rcvd_idx]) == 3:
                info = self._task_infos.pop(self._rcvd_idx)
                self._structure_infos.append(info[2])
                return info[1]

            try:
                # [ avoid hang ]: main process may blocking at _reader.read_next when
                # KeyboardInterrupt, we do following tradeoff:
                # 1. get data with timeout, MP_STATUS_CHECK_INTERVAL(5s) as timeout
                #    default, if KeyboardInterrupt blocking, failed workers will be
                #    checked and raise RuntimeError to quit DataLoader in timeout
                #    exception handling.
                # 2. if get data timeout and check workers all alive, continue to
                #    get data again
                data = self._data_queue.get(timeout=self._timeout)
            except Exception as e:
                # check if thread done event set when waiting data
                if self._thread_done_event.is_set():
                    continue

                # check failed workers
                failed_workers = []
                for i, w in enumerate(self._workers):
                    if self._worker_status[i] and not w.is_alive():
                        failed_workers.append(w)
                        self._shutdown_worker(i)
                if len(failed_workers) > 0:
                    self._exit_thread_unexpectedly()
                    pids = ', '.join(str(w.pid) for w in failed_workers)
                    raise RuntimeError("DataLoader {} workers exit unexpectedly, " \
                                "pids: {}".format(len(failed_workers), pids))

                # get(timeout) will call _poll(timeout) and may raise IOError
                if isinstance(e, queue.Empty) or isinstance(e, IOError):
                    # continue on timeout to keep getting data from queue
                    continue

                self._exit_thread_unexpectedly()
                logging.error("DataLoader reader thread failed({}) to read data from " \
                              "workers' result queue.".format(e))
                six.reraise(*sys.exc_info())
            else:
                if self._dataset_kind == _DatasetKind.ITER and isinstance(
                        data, _IterableDatasetStopIteration):
                    # if a worker get StopIteraion, we shutdown this worker,
                    # note that this batch indices to trigger StopIteration
                    # is discard, outstanding batch number should be decrease
                    # and another indices should be put for other workers
                    # may still working.
                    self._shutdown_worker(data.worker_id)
                    self._batches_outstanding -= 1
                    self._try_put_indices()
                    continue

                idx, batch, structure = data
                if isinstance(batch, _WorkerException):
                    self._exit_thread_unexpectedly()
                    batch.reraise()

                if idx == self._rcvd_idx:
                    del self._task_infos[idx]
                    self._structure_infos.append(structure)
                    return batch
                else:
                    self._task_infos[idx] += (batch, structure)
                    continue

    def _try_put_indices(self):
        assert self._batches_outstanding <= self._outstanding_capacity, \
                    "too many indices have been put to queue"
        # In multi-process mode for IterableDataset, _try_put_indices will
        # be called both in main process(for our implement has blocking queue,
        # and blocking queue read is in main process) and thread, which may
        # cause error following error
        #   1. "ValueError: generator already executing" in next(self._sampler_iter)
        #   2. re-enter in increase _send_idx
        # add a lock for threading save, for _try_put_indices is only a slight
        # function which is not in data reading pipeline, this lock almost no
        # influence on performance
        with self._thread_lock:
            try:
                indices = next(self._sampler_iter)
            except StopIteration:
                return

            for i in range(self._num_workers):
                worker_idx = next(self._workers_idx_cycle)
                if self._worker_status[worker_idx]:
                    break
            else:
                return

            self._indices_queues[worker_idx].put((self._send_idx, indices))
            self._task_infos[self._send_idx] = (worker_idx, )
            self._batches_outstanding += 1
            self._send_idx += 1

    def __del__(self):
        self._try_shutdown_all()

    def _shutdown_on_exit(self):
        self._try_shutdown_all(1)

    def __next__(self):
        try:
            # _batches_outstanding here record the total batch data number
            # in 'from after _try_put_indices to beforeoutput data', this
            # value should be _outstanding_capacity if data is not drained,
            # if _batches_outstanding is less than _places number, there are
            # no enough data to generate next output, close blocking_queue and
            # set _thread_done_event here, py_reader will raise StopIteration,
            # end workers and indices_queues in StopIteration handling
            if self._batches_outstanding < len(self._places):
                self._thread_done_event.set()
                self._blocking_queue.close()

            if in_dygraph_mode():
                data = self._reader.read_next_var_list()
                # print(data.shape)
                data = _restore_batch(data, self._structure_infos.pop(0))
            else:
                if self._return_list:
                    data = self._reader.read_next_list()
                    data = [
                        _restore_batch(d, s)
                        for d, s in zip(data, self._structure_infos[:len(
                            self._places)])
                    ]
                    self._structure_infos = self._structure_infos[len(
                        self._places):]
                    # static graph organized data on multi-device with list, if
                    # place number is 1, there is only 1 device, extra the data
                    # from list for devices to be compatible with dygraph mode
                    if len(self._places) == 1:
                        data = data[0]
                else:
                    data = self._reader.read_next()
            self._on_output_batch()
            return data
        except StopIteration:
            self._reader.shutdown()
            self._try_shutdown_all()
            six.reraise(*sys.exc_info())

    # python2 compatibility
    def next(self):
        return self.__next__()

    def _on_output_batch(self):
        for _ in range(len(self._places)):
            self._batches_outstanding -= 1
            self._try_put_indices()

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
                 batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, collate_fn=collate_fn, num_workers=num_workers, 
                 use_buffer_reader=use_buffer_reader, use_shared_memory=use_shared_memory, timeout=timeout, worker_init_fn=worker_init_fn)

        self.scale = args.scale

    def __iter__(self):
        if self.num_workers >= 1: 
            return _MSDataLoaderIter(self)
        else:
            return _DataLoaderIterSingleProcess(self)
