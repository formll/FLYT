# Filter Like You Test: Data-Driven Data Filtering for CLIP Pretraining

This repository contains the implementation of the algorithms in the paper [Filter Like You Test: Data-Driven Data Filtering for CLIP Pretraining](https://arxiv.org/abs/2503.08805) by Mikey Shechter and Yair Carmon. We thank [OpenCLIP](https://github.com/mlfoundations/open_clip) for open-sourcing their code. To simplify usage, we've incorporated their codebase with our modifications. For details, see [Changes in open_clip](#changes-in-open_clip).

## Data and Models

Our models and M-FLYT input scores are available in this [Hugging Face collection](https://huggingface.co/collections/formll/flyt-67bb167366ec0fa0d5b8e4bd). Specifically: 

- FLYT and M-FLYT scoring models, as well as models trained on datasets filtered by these methods, can be found in [formll/FLYT-models](https://huggingface.co/formll/FLYT-models). 
- The M-FLYT input scores, formatted as a parquet dataset, are available in [formll/M-FLYT-input-scores](https://huggingface.co/datasets/formll/M-FLYT-input-scores)
- The models we used to generate these scores can be found in [formll/M-FLYT-input-scores-models](https://huggingface.co/formll/M-FLYT-input-scores-models)

## Usage

```bash
python -m venv .env
source .env/bin/activate
pip install -r requirements.txt
```

An example command for training M-FLYT:

```bash
python -m FLYT.train_flyt \
    --upstream_data_dir ${UPSTREAM_DATA_DIR} \
    --downstream_data_dir ${DOWNSTREAM_DATA_DIR} \
    --datacomp_eval_dir ${DATACOMP_EVAL_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --exp_name ${EXP_NAME} \
    --precision ${PRECISION} \
    --num_checkpoints ${NUM_CHECKPOINTS} \
    --save_frequency 1 \
    --seed ${SEED} \
    --report_to_wandb \
    --accum_freq 1 \
    --wandb_project_name ${WANDB_PROJECT_NAME} \
    --log_every_n_steps 1 \
    --downstream_task_names imagenet \
    --model ViT-B-32 \
    --reference_learning_rate 5e-5 \
    --scoring_learning_rate 1e-3 \
    --warmup 100 \
    --upstream_batch_size 4096 \
    --downstream_batch_size 3072 \
    --n_iterations 5000 \
    --reference_pretrained openai \
    --scoring_pretrained openai \
    --update_reference_model \
    --downstream_logit_scale 2.65926
```

Notable arguments:

- `--downstream_data_dir` should point to a parent directory containing downstream tasks training data. See `flyt.data.UpAndDownstreamDataSet` for implementation details.
- `--datacomp_eval_dir` should point to a directory containing the classnames.txt and zeroshot_classification_templates.txt files for each downstream dataset. These can be obtained by downloading the DataComp eval datasets (see <https://github.com/mlfoundations/datacomp?tab=readme-ov-file#evaluation>).

An implementation of SCS, HCS, and FLYT model loading can be found in the notebook `examples.ipynb`

## Repository structure

- `train_flyt.py`: FLYT training entry point (inspired by [datacomp/train.py](https://github.com/mlfoundations/datacomp/blob/main/train.py)).
- `flyt`: Directory containing our implementation code.
- `open_clip`: Contains a modified copy of open_clip with changes as detailed below.

### Changes in open_clip

Instead of rewriting existing functionality, we modified the OpenCLIP codebase where needed. We chose to organize FLYT-specific implementations in the separate `flyt` directory to make it easier to distinguish between original OpenCLIP code and our additions. The training script (`train_flyt.py`) calls `open_clip.src.open_clip_train.main.main`, which depends on both the modified OpenCLIP and the `flyt` directory.

Specific changes in OpenCLIP:

- `open_clip_train.main.main`: Modified to execute FLYT-specific logic when the `args.flyt` flag is present.
- `open_clip_train.params`: Added FLYT-specific parameters.
- `open_clip_train.data`: Added dataset_weighted parameter to ResampledShards2 and expand_urls to enable balanced downstream dataset weighting.
- `open_clip.factory.create_model`: Added downstream_logit_scale parameter (downstream temperature).
- `open_clip.loss.ClipLoss`: Updated to use our custom AllGather implementation that's compatible with torch.func.

## Citation
```
@article{shechter2025filter,
  title={Filter Like You Test: Data-Driven Data Filtering for {CLIP} Pretraining}, 
  author={Mikey Shechter and Yair Carmon}, 
  journal={arXiv:2503.08805}, 
  year={2025},
}  
```
