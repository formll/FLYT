import os
import math
import logging
import torch
import webdataset as wds
import pandas as pd
from functools import lru_cache

from open_clip.src.open_clip import get_input_dtype
from open_clip.src.open_clip_train.data import (
    SharedEpoch, DataInfo, ResampledShards2, tarfile_to_samples_nothrow, log_and_continue, filter_no_caption_or_no_image,
    _SAMPLE_SHUFFLE_SIZE, _SAMPLE_SHUFFLE_INITIAL)
from open_clip.src.open_clip_train.distributed import is_master


class ParquetReader:
    """Parquet reader for M-FLYT input scores."""
    def __init__(self, parquet_dir, fields, input_dtype):
        self.parquet_dir = parquet_dir
        self.fields = fields
        self.input_dtype = input_dtype
        
    @lru_cache(1000)
    def read_parquet_chunk(self, shard_id):
        parquet_file = os.path.join(self.parquet_dir, f"{shard_id}_scores.parquet")
        df = pd.read_parquet(parquet_file, columns=['key'] + self.fields)
        return df.set_index('key')

    def __call__(self, sample):
        shard_id = sample['__key__'][:8]
        df_chunk = self.read_parquet_chunk(shard_id)
        
        try:
            row = df_chunk.loc[sample['__key__']]
            all_fields = []
            for field in self.fields:
                all_fields.append(torch.tensor([row[field]])[0])
                
        except KeyError:
            print(f"Key not found in parquet data: {sample['key']}")
        
        all_fields = torch.stack(all_fields).to(dtype=self.input_dtype)
        sample["scores"] = all_fields
        return sample


def get_wds_actual_dataset(
    input_shards,
    args,
    preprocess_img,
    batch_size,
    shared_epoch=0,
    weights=None,
    tokenizer=None,
    load_classes=False,
    parquet_dir=None,
    parquet_fields=None,
    dataset_weighted=False):
    """Returns a wds Dataset object (unlike open_clip_train.data.get_wds_dataset which returns a DataInfo object)."""
    input_dtype = get_input_dtype(args.precision)
    
    if isinstance(shared_epoch, int):
        shared_epoch = SharedEpoch(epoch=shared_epoch)  # create a shared epoch store to sync epoch to dataloader worker proc
    
    # We always use resampled
    pipeline = [ResampledShards2(
        input_shards,
        weights=weights,
        deterministic=True,
        epoch=shared_epoch,
        dataset_weighted=dataset_weighted
    )]

    shuffle_buf_size = 50_000 if args.downstream_clip_loss else _SAMPLE_SHUFFLE_SIZE
    shuffle_initial_size = 10_000 if args.downstream_clip_loss else _SAMPLE_SHUFFLE_INITIAL
        
    pipeline.extend([
        tarfile_to_samples_nothrow,
        wds.shuffle(
            bufsize=shuffle_buf_size,
            initial=shuffle_initial_size,
        ),
    ])
    
    pipeline.extend([
        wds.select(filter_no_caption_or_no_image),
        wds.decode("pilrgb", handler=log_and_continue),
        wds.rename(image="jpg;png;jpeg;webp", text="txt"),
        wds.map_dict(image=preprocess_img, text=lambda text: tokenizer(text)[0])
    ])

    # This elif is possible because we only use parquet_dir for upstream data and never load_classes for upstream data.
    if load_classes:
        pipeline.extend([
            wds.rename(label="label;cls"),
            wds.map_dict(label=lambda label: torch.tensor(int(label), dtype=torch.int32)),
            wds.to_tuple("image", "text", "label")
        ])
    elif parquet_dir is not None:
        parquet_reader = ParquetReader(parquet_dir, parquet_fields, input_dtype)
        pipeline.extend([
            wds.map(parquet_reader),
            wds.to_tuple("image", "text", "scores")
        ])
    else:
        pipeline.extend([
            wds.to_tuple("image", "text")
        ])
    
    pipeline.extend([
        wds.batched(batch_size, partial=False)
    ])

    dataset = wds.DataPipeline(*pipeline)
    return dataset


class UpAndDownstreamDataSet(wds.DataPipeline):
    def __init__(self, args, preprocess_train, shared_epoch, tokenizer, load_classes=False, parquet_dir=None, parquet_fields=None):
        task_list = {
            "imagenet": os.path.join(args.downstream_data_dir, "imagenet_oai_templates", "{00000000..00012811}.tar"),
            "mnist": os.path.join(args.downstream_data_dir, "mnist", "{00000000..00000599}.tar"),
            "svhn": os.path.join(args.downstream_data_dir, "svhn_cropped", "{00000000..00000732}.tar"),
            "cifar10": os.path.join(args.downstream_data_dir, "cifar10", "{00000000..00000499}.tar"),
            "cifar100": os.path.join(args.downstream_data_dir, "cifar100", "{00000000..00000499}.tar"),
            "food101": os.path.join(args.downstream_data_dir, "food-101", "{00000000..00000757}.tar"),
            "flowers102": os.path.join(args.downstream_data_dir, "flowers102", "{00000000..00000010}.tar"),
            "pet": os.path.join(args.downstream_data_dir, "pet", "{00000000..00000036}.tar"),
            "iwildcam": os.path.join(args.downstream_data_dir, "iwildcam", "{00000000..00001298}.tar"),
            "sun397": os.path.join(args.downstream_data_dir, "sun397", "{00000000..00000761}.tar"),
            "eurosat": os.path.join(args.downstream_data_dir, "eurosat", "{00000000..00000161}.tar"),
            "cars": os.path.join(args.downstream_data_dir, "cars", "{00000000..00000081}.tar"),
            "dtd": os.path.join(args.downstream_data_dir, "dtd", "{00000000..00000018}.tar"),
            "resisc45": os.path.join(args.downstream_data_dir, "resisc45", "{00000000..00000188}.tar"),
            "gtsrb": os.path.join(args.downstream_data_dir, "gtsrb", "{00000000..00000392}.tar"),
            "aircraft": os.path.join(args.downstream_data_dir, "aircraft", "{00000000..00000066}.tar"),
            "voc": os.path.join(args.downstream_data_dir, "voc", "{00000000..00000099}.tar"),
            "country": os.path.join(args.downstream_data_dir, "country", "{00000000..00000316}.tar"),
            "rendered_sst2": os.path.join(args.downstream_data_dir, "rendered_sst2", "{00000000..00000069}.tar"),
            "fmow": os.path.join(args.downstream_data_dir, "fmow", "{00000000..00000768}.tar"),
            "dollar_street": os.path.join(args.downstream_data_dir, "dollar_street", "{00000000..00000145}.tar"),
            "stl10": os.path.join(args.downstream_data_dir, "stl10", "{00000000..00000049}.tar"),
        }
        self.shared_epoch = shared_epoch
        self.args = args
        
        self.upstream_dataset = get_wds_actual_dataset(
            args.train_data,
            args,
            preprocess_train,
            batch_size=args.batch_size,
            shared_epoch=self.shared_epoch,
            weights=args.train_data_upsampling_factors,
            tokenizer=tokenizer,
            parquet_dir=parquet_dir,
            parquet_fields=parquet_fields)
        if is_master(args):
            logging.info(f"Upstream data: {args.train_data}")
        
        self.downstream_datasets = list()
        downstream_task_names = args.downstream_task_names.split('::')
        
        downstream_batch_size = args.downstream_batch_size if args.downstream_batch_size is not None else args.batch_size
        downstream_dataset_kwargs = {
            "args": args,
            "preprocess_img": preprocess_train,
            "batch_size": downstream_batch_size, 
            "shared_epoch": self.shared_epoch, 
            "tokenizer": tokenizer, 
            "load_classes": load_classes
        }
        if args.downstream_clip_loss:
            datapath = "::".join([task_list[downstream_task_name] for downstream_task_name in downstream_task_names])
            print_paths = '\n'.join(datapath.split('::'))
            if is_master(args):
                logging.info(f"Loading {args.downstream_task_names} from \n{print_paths}")
            # Using CLIP loss we allow downstream_data_upsampling_factors because many downstream datasets can be loaded to each dataloader.
            downstream_data = get_wds_actual_dataset(
                datapath, 
                weights=args.downstream_data_upsampling_factors,
                dataset_weighted=args.dataset_weighted,
                **downstream_dataset_kwargs)
            self.downstream_datasets.append(downstream_data)
        else:
            for downstream_task_name in downstream_task_names:
                if is_master(args):
                    logging.info(f"Loading {downstream_task_name} from {task_list[downstream_task_name]}")
                # Using CE loss we don't allow downstream_data_upsampling_factors since each dataset is loaded separately.
                downstream_data = get_wds_actual_dataset(task_list[downstream_task_name], **downstream_dataset_kwargs)
                self.downstream_datasets.append(downstream_data)
        
        self.upstream_dataiter = iter(self.upstream_dataset)
        self.downstream_dataiters = [iter(dl) for dl in self.downstream_datasets]
        
        # datacomp divides so args.batch_size is divided args.world_size
        self.global_batch_size = args.batch_size * args.world_size

        global_num_batches = math.ceil(args.train_num_samples / self.global_batch_size)  # Total number of batches on all gpus
        num_workers = max(1, args.workers)
        self.num_batches_per_worker = math.ceil(global_num_batches / num_workers)  # per dataloader worker
        self.num_batches = self.num_batches_per_worker * num_workers
        self.num_samples = self.num_batches * self.global_batch_size        

    def __len__(self):
        return self.num_batches_per_worker

    def __iter__(self):
        for _ in range(len(self)):
            try:
                upstream_batch = next(self.upstream_dataiter)
            except StopIteration:
                # refresh dataiter if dataloader is used up.
                self.upstream_dataiter = iter(self.upstream_dataloader)
                upstream_batch = next(self.upstream_dataiter)
                
            downstream_batches = list()
            for i in range(len(self.downstream_dataiters)):
                try:
                    downstream_batches.append(next(self.downstream_dataiters[i]))
                except StopIteration:
                    # refresh dataiter if dataloader is used up.
                    self.downstream_dataiters[i] = iter(self.downstream_dataloaders[i])
                    downstream_batches.append(next(self.downstream_dataiters[i]))
            
            batch = (*upstream_batch, downstream_batches)

            yield batch


def get_up_and_downstream_datainfo(args, preprocess_train, epoch, tokenizer, load_classes=False, parquet_dir=None, parquet_fields=None):
    shared_epoch = SharedEpoch(epoch=epoch)
    dataset = UpAndDownstreamDataSet(args, preprocess_train, shared_epoch, tokenizer, load_classes=load_classes, parquet_dir=parquet_dir, parquet_fields=parquet_fields)
    n_downstream_datasets = len(dataset.downstream_datasets)
    dataset = dataset.with_epoch(dataset.num_batches_per_worker)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=n_downstream_datasets < 5  # If there are many datasets, restart workers each epoch. This is a hack for fixing some RAM memory leak or something that looks like a memory leak.
    )

    dataloader.num_batches = dataset.num_batches
    dataloader.num_samples = dataset.num_samples
    dataloader.num_batches_per_worker = dataset.num_batches_per_worker
    if is_master(args):
        logging.info(f"Inner data loader {dataset.num_batches=}")
        logging.info(f"Inner data loader {dataset.num_samples=}")
    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)


def get_flyt_data(args, preprocess_fns, epoch=0, tokenizer=None, parquet_dir=None, parquet_fields=None):
    preprocess_train, _ = preprocess_fns
    data = {}
    data['train'] = get_up_and_downstream_datainfo(args, preprocess_train, epoch, tokenizer, load_classes=True, parquet_dir=parquet_dir, parquet_fields=parquet_fields)
    return data
