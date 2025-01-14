<div align="center">

# 🔥 Flame: Flash Linear Attention Made Easy

A minimal, efficient training framework for Flash Linear Attention models

</div>

## Usage

To get started, run `bash train.sh -h` to see all available command line options. 
Here are some of the most important options you'll want to configure:

```bash
options:
  -h, --help            show this help message and exit
  --job.config_file JOB.CONFIG_FILE
                        Job config file
  --job.dump_folder JOB.DUMP_FOLDER
                        Folder to dump job outputs
  --job.description JOB.DESCRIPTION
                        Description of the job
  --job.use_for_integration_test
                        Add this config to the integration test suite
  --job.print_args      Print the args to terminal
  --model.config MODEL.CONFIG
                        Path to the model config
  --model.norm_type MODEL.NORM_TYPE
                        Type of layer normalization to use [layernorm,
                        np_layernorm, rmsnorm, fused_rmsnorm]
  --model.tokenizer_path MODEL.TOKENIZER_PATH
                        Tokenizer path
  --profiling.enable_profiling
                        Whether to enable pytorch profiler
  --profiling.save_traces_folder PROFILING.SAVE_TRACES_FOLDER
                        Trace files location
  --profiling.profile_freq PROFILING.PROFILE_FREQ
                        How often to collect profiler traces, in iterations
  --profiling.enable_memory_snapshot
                        Whether to dump memory snapshot
  --profiling.save_memory_snapshot_folder PROFILING.SAVE_MEMORY_SNAPSHOT_FOLDER
                        Memeory snapshot files location
  --optimizer.name OPTIMIZER.NAME
                        Optimizer to use
  --optimizer.lr OPTIMIZER.LR
                        Learning rate to use
  --optimizer.fused     Whether the fused implementation(CUDA only) is used.
  --optimizer.scheduler {wsd,cosine,linear}
                        Scheduler to use. Currently supported: wsd, cosine,
                        and linear.
  --optimizer.early_step_in_backward
                        Whether to apply optimizer in the backward. Caution,
                        optimizer_in_backward is not compatible with gradients
                        clipping, users should not call
                        register_post_accumulate_grad_hook after the optimizer
                        is built.
  --training.batch_size TRAINING.BATCH_SIZE
                        Batch size
  --training.seq_len TRAINING.SEQ_LEN
                        Sequence length
  --training.warmup_steps TRAINING.WARMUP_STEPS
                        Steps for lr scheduler warmup, normally 1/5 of
                        --training.steps
  --training.gradient_accumulation_steps TRAINING.GRADIENT_ACCUMULATION_STEPS
                        Number of steps to accumulate gradients before
                        updating parameters
  --training.steps TRAINING.STEPS
                        How many train steps to run
  --training.max_norm TRAINING.MAX_NORM
                        Max norm for gradient clipping
  --training.skip_nan_inf
                        Skip batch updates when NaN or INF gradients are
                        encountered during training
  --training.dataset TRAINING.DATASET
                        Dataset to use
  --training.dataset_name TRAINING.DATASET_NAME
                        The name of the dataset config
  --training.dataset_split TRAINING.DATASET_SPLIT
                        Dataset split to use
  --training.num_workers TRAINING.NUM_WORKERS
                        Number of subprocesses to use for data loading. 0
                        means that the data will be loaded in the main
                        process.
  --training.prefetch_factor TRAINING.PREFETCH_FACTOR
                        Number of batches loaded in advance by each worker.2
                        means there will be a total of 2 * num_workers batches
                        prefetched across all workers.
  --training.data_parallel_replicate_degree TRAINING.DATA_PARALLEL_REPLICATE_DEGREE
                        The `data_parallel_replicate_degree` argument
                        specifies the degree of data parallelism for weight
                        replication. When this value is greater than 1,
                        weights will be replicated across
                        `data_parallel_replicate_degree` ranks. If
                        `data_parallel_shard_degree` is also greater than 1,
                        the parallelism method used is HSDP (Hybrid Sharded
                        Data Parallelism). Otherwise, the parallelism method
                        used is DDP (Distributed Data Parallelism). 1 means
                        disabled.
  --training.data_parallel_shard_degree TRAINING.DATA_PARALLEL_SHARD_DEGREE
                        The `data_parallel_shard_degree` argument specifies
                        the degree of data parallelism for weight sharding.
                        When this value is greater than 1, weights will be
                        sharded across `data_parallel_shard_degree` ranks. If
                        `data_parallel_replicate_degree` is also greater than
                        1, the parallelism method used is HSDP (Hybrid Sharded
                        Data Parallelism). Otherwise, the parallelism method
                        used is FSDP (Fully Sharded Data Parallelism). -1
                        means leftover ranks will be used (After
                        DP_REPLICATE/SP/PP). Note that only
                        `data_parallel_shard_degree` can be negative. 1 means
                        disabled.
  --training.enable_cpu_offload
                        Whether to apply CPU offloading of parameters,
                        gradients, and optimizer states in FSDP
  --training.tensor_parallel_degree TRAINING.TENSOR_PARALLEL_DEGREE
                        Tensor Parallelism degree. 1 means disabled.
  --training.disable_loss_parallel
                        Whether to apply loss parallel when sequence parallel
                        is enabled
  --training.mixed_precision_param {bfloat16,float32}
                        torch dtype to use for parameters when applying mixed
                        precision via FSDP. This feature only takes effect
                        when data_parallel_shard_degree > 1
  --training.mixed_precision_reduce {float32}
                        torch dtype to use for reductions when applying mixed
                        precision via FSDP. This feature only takes effect
                        when data_parallel_shard_degree > 1
  --training.compile    Whether to compile the model
  --training.gc_freq TRAINING.GC_FREQ
                        Python garbage control scheduling interval, in steps
  --training.seed TRAINING.SEED
                        Choose the base RNG seed used for training
  --training.deterministic
                        Use deterministic algorithms wherever possible, may be
                        slower
  --metrics.log_freq METRICS.LOG_FREQ
                        How often to log metrics to TensorBoard, in iterations
  --metrics.enable_tensorboard
                        Whether to log metrics to TensorBoard
  --metrics.disable_color_printing
                        Whether to disable color printing in logs
  --metrics.save_tb_folder METRICS.SAVE_TB_FOLDER
                        Folder to dump TensorBoard states
  --metrics.rank_0_only
                        Whether to save TensorBoard metrics only for rank 0 or
                        for all ranks. When pipeline_parallel_degree is > 1,
                        this option uses the 0th rank of the last stage
                        pipeline group, which is the only stage that computes
                        loss metrics.
  --metrics.enable_wandb
                        Whether to log metrics to Weights & Biases
  --checkpoint.enable_checkpoint
                        Whether to enable checkpoint
  --checkpoint.folder CHECKPOINT.FOLDER
                        The folder to store the checkpoints. When
                        enable_checkpoint is set to true, checkpoints will be
                        in {--job.dump_folder}/{--checkpoint.folder}.
  --checkpoint.interval_type CHECKPOINT.INTERVAL_TYPE
                        Checkpointing interval unit of measurement ['step',
                        'seconds']
  --checkpoint.interval CHECKPOINT.INTERVAL
                        Checkpointing interval, in steps or seconds depending
                        on --checkpoint.interval_type
  --checkpoint.model_weights_only
                        When model_weights_only=True, only model weights will
                        be saved at the end of training. With this,
                        checkpoints can be loaded using `torch.load(...,
                        weights_only=True)` after conversion. When
                        model_weights_only=False, the full checkpoint will be
                        saved. A full checkpoint includes model, optimizer and
                        train_state, which can be used to resume training. The
                        default value is false.
  --checkpoint.export_dtype {float16,bfloat16,float32}
                        Converts to the specified precision when training
                        completes and model_weights_only=true. Currently
                        supports float32, float16, and bfloat16. The default
                        value is float32.
```
