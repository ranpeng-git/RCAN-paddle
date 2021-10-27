from importlib import import_module

from dataloader import MSDataLoader
# from paddle.io.dataloader import default_collate
from paddle.io import DataLoader

class Data:
    def __init__(self, args):
        # kwargs = {}
        # if not args.cpu:
        #     kwargs['collate_fn'] = default_collate
        #     kwargs['pin_memory'] = True
        # else:
        #     kwargs['collate_fn'] = default_collate
        #     kwargs['pin_memory'] = False

        self.loader_train = None
        if not args.test_only:
            module_train = import_module('data.' + args.data_train.lower())
            trainset = getattr(module_train, args.data_train)(args)
            # self.loader_train = MSDataLoader(
            #     args,
            #     trainset,
            #     batch_size=args.batch_size,
            #     shuffle=True,
            #     # **kwargs
            # )
            self.loader_train = MSDataLoader(args , 
                trainset, feed_list=None, places=None, return_list=False, batch_sampler=None,
                 batch_size=args.batch_size, shuffle=True, drop_last=False, collate_fn=None, num_workers=args.n_threads, 
                 use_buffer_reader=True, use_shared_memory=True, timeout=0, worker_init_fn=None
            )

        if args.data_test in ['Set5', 'Set14', 'B100', 'Urban100']:
            if not args.benchmark_noise:
                module_test = import_module('data.benchmark')
                testset = getattr(module_test, 'Benchmark')(args, train=False)
            else:
                module_test = import_module('data.benchmark_noise')
                testset = getattr(module_test, 'BenchmarkNoise')(
                    args,
                    train=False
                )

        else:
            module_test = import_module('data.' +  args.data_test.lower())
            testset = getattr(module_test, args.data_test)(args, train=False)

        # self.loader_test = MSDataLoader(
        #     args,
        #     testset,
        #     batch_size=1,
        #     shuffle=False,
        #     # **kwargs
        # )
        self.loader_test = MSDataLoader(args , 
            testset, feed_list=None, places=None, return_list=False, batch_sampler=None,
                 batch_size=1, shuffle=False, drop_last=False, collate_fn=None, num_workers=args.n_threads, 
                 use_buffer_reader=True, use_shared_memory=True, timeout=0, worker_init_fn=None
        )
