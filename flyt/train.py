import os
import logging
import math
import random
import time
import torch
import torch.nn.functional as F

try:
    import wandb
except ImportError:
    wandb = None
    
from functools import lru_cache
from torch.func import functional_call, grad, grad_and_value

from open_clip.src.open_clip import get_tokenizer, get_input_dtype
from open_clip.src.open_clip_train.precision import get_autocast
from open_clip.src.open_clip_train.train import AverageMeter
from open_clip.src.open_clip_train.distributed import is_master

from flyt.distributed import apply_all_gather


@lru_cache()
def get_classnames(datacomp_eval_dir, rank, world_size, downstream_task_name):
    task_classnames_list = {
        "imagenet": os.path.join(datacomp_eval_dir, "wds_imagenet1k_test", "classnames.txt"),
        "mnist": os.path.join(datacomp_eval_dir, "wds_mnist_test", "classnames.txt"),
        "svhn": os.path.join(datacomp_eval_dir, "wds_vtab-svhn_test", "classnames.txt"),
        "cifar10": os.path.join(datacomp_eval_dir, "wds_cifar10_test", "classnames.txt"),
        "cifar100": os.path.join(datacomp_eval_dir, "wds_vtab-cifar100_test", "classnames.txt"),
        "food101": os.path.join(datacomp_eval_dir, "wds_food101_test", "classnames.txt"),
        "flowers102": os.path.join(datacomp_eval_dir, "wds_vtab-flowers_test", "classnames.txt"),
        "pet": os.path.join(datacomp_eval_dir, "wds_vtab-pets_test", "classnames.txt"),
        "iwildcam": os.path.join(datacomp_eval_dir, "wds_wilds-iwildcam_test", "classnames.txt"),
        "sun397": os.path.join(datacomp_eval_dir, "wds_sun397_test", "classnames.txt"),
        "eurosat": os.path.join(datacomp_eval_dir, "wds_vtab-eurosat_test", "classnames.txt"),
        "cars": os.path.join(datacomp_eval_dir, "wds_cars_test", "classnames.txt"),
        "dtd": os.path.join(datacomp_eval_dir, "wds_vtab-dtd_test", "classnames.txt"),
        "resisc45": os.path.join(datacomp_eval_dir, "wds_vtab-resisc45_test", "classnames.txt"),
        "gtsrb": os.path.join(datacomp_eval_dir, "wds_gtsrb_test", "classnames.txt"),
        "aircraft": os.path.join(datacomp_eval_dir, "wds_fgvc_aircraft_test", "classnames.txt"),
        "voc": os.path.join(datacomp_eval_dir, "wds_voc2007_test", "classnames.txt"),
        "country": os.path.join(datacomp_eval_dir, "wds_country211_test", "classnames.txt"),
        "rendered_sst2": os.path.join(datacomp_eval_dir, "wds_renderedsst2_test", "classnames.txt"),
        "fmow": os.path.join(datacomp_eval_dir, "wds_wilds-fmow_test", "classnames.txt"),
        "dollar_street": os.path.join(datacomp_eval_dir, "wds_dollar_street_test", "classnames.txt"),
        "stl10": os.path.join(datacomp_eval_dir, "wds_stl10_test", "classnames.txt"),
    }
    
    classnames_path = task_classnames_list[downstream_task_name]
    with open(classnames_path, "r") as f:
        classnames = [c.strip() for c in f.readlines()]
        
    n_classes_per_rank = math.ceil(len(classnames) / world_size)
    classnames = classnames[n_classes_per_rank * rank:n_classes_per_rank * (rank + 1)]
    return classnames


@lru_cache()
def get_templates(datacomp_eval_dir, downstream_task_name):
    task_templates_list = {
        "imagenet": os.path.join(datacomp_eval_dir, "wds_imagenet1k_test", "zeroshot_classification_templates.txt"),
        "mnist": os.path.join(datacomp_eval_dir, "wds_mnist_test", "zeroshot_classification_templates.txt"),
        "svhn": os.path.join(datacomp_eval_dir, "wds_vtab-svhn_test", "zeroshot_classification_templates.txt"),
        "cifar10": os.path.join(datacomp_eval_dir, "wds_cifar10_test", "zeroshot_classification_templates.txt"),
        "cifar100": os.path.join(datacomp_eval_dir, "wds_vtab-cifar100_test", "zeroshot_classification_templates.txt"),
        "food101": os.path.join(datacomp_eval_dir, "wds_food101_test", "zeroshot_classification_templates.txt"),
        "flowers102": os.path.join(datacomp_eval_dir, "wds_vtab-flowers_test", "zeroshot_classification_templates.txt"),
        "pet": os.path.join(datacomp_eval_dir, "wds_vtab-pets_test", "zeroshot_classification_templates.txt"),
        "iwildcam": os.path.join(datacomp_eval_dir, "wds_wilds-iwildcam_test", "zeroshot_classification_templates.txt"),
        "sun397": os.path.join(datacomp_eval_dir, "wds_sun397_test", "zeroshot_classification_templates.txt"),
        "eurosat": os.path.join(datacomp_eval_dir, "wds_vtab-eurosat_test", "zeroshot_classification_templates.txt"),
        "cars": os.path.join(datacomp_eval_dir, "wds_cars_test", "zeroshot_classification_templates.txt"),
        "dtd": os.path.join(datacomp_eval_dir, "wds_vtab-dtd_test", "zeroshot_classification_templates.txt"),
        "resisc45": os.path.join(datacomp_eval_dir, "wds_vtab-resisc45_test", "zeroshot_classification_templates.txt"),
        "gtsrb": os.path.join(datacomp_eval_dir, "wds_gtsrb_test", "zeroshot_classification_templates.txt"),
        "aircraft": os.path.join(datacomp_eval_dir, "wds_fgvc_aircraft_test", "zeroshot_classification_templates.txt"),
        "voc": os.path.join(datacomp_eval_dir, "wds_voc2007_test", "zeroshot_classification_templates.txt"),
        "country": os.path.join(datacomp_eval_dir, "wds_country211_test", "zeroshot_classification_templates.txt"),
        "rendered_sst2": os.path.join(datacomp_eval_dir, "wds_renderedsst2_test", "zeroshot_classification_templates.txt"),
        "fmow": os.path.join(datacomp_eval_dir, "wds_wilds-fmow_test", "zeroshot_classification_templates.txt"),
        "dollar_street": os.path.join(datacomp_eval_dir, "wds_dollar_street_test", "zeroshot_classification_templates.txt"),
        "stl10": os.path.join(datacomp_eval_dir, "wds_stl10_test", "zeroshot_classification_templates.txt"),
    }
    
    templates_path = task_templates_list[downstream_task_name]
    with open(templates_path, "r") as f:
        templates = [t.strip() for t in f.readlines()]
    return templates


@lru_cache()
def get_cached_tokenizer(model_type):
    return get_tokenizer(model_type)


def sample_downstream_texts(args, device, downstream_task_name):
    classnames = get_classnames(args.datacomp_eval_dir, args.rank, args.world_size, downstream_task_name)
    templates = get_templates(args.datacomp_eval_dir, downstream_task_name)
    tokenizer = get_cached_tokenizer(args.model)
    random_templates = random.choices(templates, k=len(classnames))
    batch_texts = []
    for t, c in zip(random_templates, range(len(classnames))):
        batch_texts.append(t.format(c=classnames[c]))
    tokenized_texts = tokenizer(batch_texts)
    tokenized_texts = tokenized_texts.to(device=device, non_blocking=True)
    return tokenized_texts


def train_one_flyt_epoch(
    reference_model, scoring_model,
    data,
    downstream_loss, weighted_loss,
    epoch,
    reference_optimizer, scoring_optimizer,
    downstream_scaler, weighted_scaler,
    scheduler, scoring_scheduler,
    args):
    def get_loss_weights(scoring_params):
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
            if args.parquet_dir is None:  # FLYT
                scores = functional_call(scoring_model, (scoring_params, scoring_buffers), (upstream_images, upstream_texts))
            else:  # M-FLYT
                scores = functional_call(scoring_model, (scoring_params, scoring_buffers), (upstream_scores,))
        
        all_scores = apply_all_gather(scores, args.distributed)
        loss_weights = F.softmax(all_scores, dim=0).squeeze()
        
        return loss_weights

    def compute_weighted_loss(reference_params, scoring_params):
        with autocast():
            with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
                model_out = functional_call(reference_model, (reference_params, reference_buffers), (upstream_images, upstream_texts))
                loss_weights = get_loss_weights(scoring_params)

            # Save detached embeddings to avoid recalculating them later
            detached_embeds['image_features'] = model_out['image_features'].detach()
            detached_embeds['text_features'] = model_out['text_features'].detach()
            detached_embeds['logit_scale'] = model_out['logit_scale'].detach()
            detached_loss_weights['loss_weights'] = loss_weights.detach()
            
            w_loss = weighted_loss(loss_weights=loss_weights, **model_out)
        scaled_loss = weighted_scaler.scale(w_loss)
        return scaled_loss
    
    def compute_l_pre_update(reference_params, downstream_task_name):
        """This is not a part of FLYT. It is here for logging the difference before and after the single reference model step."""
        with torch.no_grad():
            with autocast():
                if args.downstream_clip_loss:
                    with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
                        down_model_out = functional_call(reference_model, (reference_params, reference_buffers), (downstream_images, downstream_texts))
                        downstream_logit_scale = 1 if args.downstream_logit_scale is None else reference_params['downstream_logit_scale'].exp()
                        down_model_out['logit_scale'] = downstream_logit_scale
                    return downstream_loss(**down_model_out)
                else:
                    sampled_texts = sample_downstream_texts(args, device, downstream_task_name)
                    with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
                        down_model_out = functional_call(reference_model, (reference_params, reference_buffers), (downstream_images, sampled_texts))
                    
                    classifier = apply_all_gather(down_model_out['text_features'], args.distributed).T
                    image_features = apply_all_gather(down_model_out['image_features'], args.distributed)
                    all_downstream_cls = apply_all_gather(downstream_cls, args.distributed)
                    
                    logits = image_features @ classifier
                    ce_loss = F.cross_entropy(logits, all_downstream_cls)
                    return ce_loss
    
    def l_hat(gradients, found_inf, reference_params, downstream_task_name, downstream_images, downstream_texts, downstream_cls):
        downstream_logit_scale = 1 if args.downstream_logit_scale is None else gradients['downstream_logit_scale'].exp()
            
        if found_inf.item() == 0:
            actual_gradients = {k: v for (k, v) in gradients.items() if k != 'downstream_logit_scale'}
            actual_reference_params = {k: v for (k, v) in reference_params.items() if k != 'downstream_logit_scale'}
                
            new_params = reference_optimizer.get_updated_params(actual_reference_params, actual_gradients)
            for p in new_params.keys():
                detached_new_params[p] = new_params[p].detach()
        else:
            logging.info(f"Found {found_inf.item()} inf or nan in l_hat. Skipping optimizer step.")
            new_params = reference_params
        weighted_scaler.update(found_inf)
        
        with autocast():
            if args.downstream_clip_loss:
                down_model_out = functional_call(reference_model, (new_params, reference_buffers), (downstream_images, downstream_texts))
                down_model_out['logit_scale'] = downstream_logit_scale
                loss = downstream_loss(**down_model_out)
            else:
                sampled_texts = sample_downstream_texts(args, device, downstream_task_name)
                if len(sampled_texts) == 0:
                    # Needed in case of very few downstream classes + CE loss. Distributed training might give a GPU 0 text examples.
                    down_model_out = functional_call(reference_model, (new_params, reference_buffers), (downstream_images, None))
                    down_model_out['text_features'] = torch.zeros((0, down_model_out['image_features'].size(1)), 
                                                                    dtype=down_model_out['image_features'].dtype, 
                                                                    device=device)
                else:
                    down_model_out = functional_call(reference_model, (new_params, reference_buffers), (downstream_images, sampled_texts))
                
                classifier = apply_all_gather(down_model_out['text_features'], args.distributed).T
                image_features = apply_all_gather(down_model_out['image_features'], args.distributed)
                all_downstream_cls = apply_all_gather(downstream_cls, args.distributed)
                
                logits = image_features @ classifier
                scaled_logits = downstream_logit_scale * logits
                loss = F.cross_entropy(scaled_logits, all_downstream_cls)

        scaled_loss = weighted_scaler.scale(loss)
        return scaled_loss
    
    def compute_v(scoring_params, reference_params):
        scaled_grads, scaled_w_loss = grad_and_value(compute_weighted_loss)(reference_params, scoring_params)
        w_loss = scaled_w_loss * weighted_scaler.inv_scale()
        
        if args.distributed:
            for k in scaled_grads.keys():
                torch.distributed.all_reduce(scaled_grads[k], op=torch.distributed.ReduceOp.SUM)
        for k in scaled_grads.keys():
            scaled_grads[k] = scaled_grads[k].detach()
            
        gradients, found_inf = weighted_scaler.unscale_grads(scaled_grads)

        total_down_found_inf = torch.tensor(0.0, dtype=torch.float32, device=found_inf.device)
        down_values = dict()
        sum_down_grads = dict()
        if args.downstream_logit_scale is not None:
            # Ugly hack, this is my way of getting a grad from l_hat to downstream_logit_scale
            gradients['downstream_logit_scale'] = reference_params['downstream_logit_scale'].detach()

        for downstream_task_name, (downstream_images, downstream_texts, downstream_cls) in zip(downstream_task_names, downstream_batches):
            scaled_down_grads, scaled_down_value = grad_and_value(l_hat)(
                gradients, found_inf, reference_params, downstream_task_name, downstream_images, downstream_texts, downstream_cls)
                        
            if args.distributed:
                for k in scaled_down_grads.keys():
                    torch.distributed.all_reduce(scaled_down_grads[k], op=torch.distributed.ReduceOp.SUM)
                    
            for k in scaled_down_grads.keys():
                scaled_down_grads[k] = scaled_down_grads[k].detach()
            
            down_gradients, down_found_inf = weighted_scaler.unscale_grads(scaled_down_grads)
            
            for k, v in down_gradients.items():
                if k not in sum_down_grads:
                    sum_down_grads[k] = v
                else:
                    sum_down_grads[k] += v
            
            total_down_found_inf += down_found_inf
            down_values[downstream_task_name] = scaled_down_value * weighted_scaler.inv_scale()
        
        total_down_gradients = {k: (v / len(downstream_task_names)).detach() for k, v in sum_down_grads.items()}
        
        if args.downstream_logit_scale is not None and found_inf.item() == 0 and total_down_found_inf.item() == 0:
            only_downstream_logit_scale_reference_params = {'downstream_logit_scale': gradients['downstream_logit_scale'].detach()}
            only_downstream_logit_scale_scaled_down_grads = {'downstream_logit_scale': total_down_gradients['downstream_logit_scale']}
            detached_new_params['downstream_logit_scale'] = reference_optimizer.get_updated_params(
                only_downstream_logit_scale_reference_params, only_downstream_logit_scale_scaled_down_grads)['downstream_logit_scale'].detach()
            
        weighted_scaler.update(total_down_found_inf)
        total_found_inf = found_inf + total_down_found_inf
        
        return total_down_gradients, down_values, total_found_inf, w_loss
        
    def loss_using_embeddings(embeds, loss_weights):
        with autocast():
            l = weighted_loss(loss_weights=loss_weights, **embeds)
        scaled_l = weighted_scaler.scale(l)
        return scaled_l
    
    def compute_h(eta, v, reference_params, loss_weights):
        new_params = dict()
        for key, p in reference_params.items():
            new_params[key] = p + (eta * v[key])

        with autocast():
            with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
                v_embeds = functional_call(reference_model, (new_params, reference_buffers), (upstream_images, upstream_texts))
        
        scaled_grad_to_embeds = grad(loss_using_embeddings)(detached_embeds, loss_weights)
        grad_to_embeds, found_inf = weighted_scaler.unscale_grads(scaled_grad_to_embeds)
        weighted_scaler.update(found_inf)
        
        if found_inf.item() != 0:
            logging.info(f"Weighted scaler found {found_inf.item()} inf or nan in gradients in compute_h. ")
            return torch.zeros((), dtype=torch.float32)
        
        image_embeds_sum = torch.einsum('ij,ij->i', grad_to_embeds['image_features'], v_embeds['image_features'])
        text_embeds_sum = torch.einsum('ij,ij->i', grad_to_embeds['text_features'], v_embeds['text_features'])
        return (image_embeds_sum + text_embeds_sum).sum()
    
    def compute_loss_to_weights(loss_weights, v, reference_params):
        h_tag = grad(compute_h)(torch.tensor(0, dtype=torch.float), v, reference_params, loss_weights)
        return h_tag
    
    def compute_loss_to_scoring_model(scoring_params_with_grad, scoring_params_without_grad, grad_to_w):
        scoring_params = {**scoring_params_with_grad, **scoring_params_without_grad}
        with autocast():
            with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
                if args.parquet_dir is None:
                    scores = functional_call(scoring_model, (scoring_params, scoring_buffers), (upstream_images, upstream_texts))
                else:
                    scores = functional_call(scoring_model, (scoring_params, scoring_buffers), (upstream_scores,))

        all_scores = apply_all_gather(scores, args.distributed)
        loss_weights = F.softmax(all_scores, dim=0)
            
        with autocast():
            loss_to_scoring_model = torch.dot(loss_weights.squeeze(), grad_to_w)
            
        scaled_loss_to_scoring_model = downstream_scaler.scale(loss_to_scoring_model)
        return scaled_loss_to_scoring_model

    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    reference_model.train()
    scoring_model.train()

    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    
    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    
    reference_params = dict(reference_model.named_parameters())
    reference_buffers = dict(reference_model.named_buffers())
    scoring_params = dict(scoring_model.named_parameters())
    scoring_buffers = dict(scoring_model.named_buffers())
    detached_new_params = dict()
    for p in reference_params.keys():
        detached_new_params[p] = reference_params[p].clone().detach()
        
    downstream_task_names = ["all_tasks"] if args.downstream_clip_loss else args.downstream_task_names.split('::')
    
    for i, batch in enumerate(dataloader):
        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        scoring_scheduler(step)
        if not args.skip_scheduler and args.update_reference_model:
            # Only schedule reference model LR if you update it
            scheduler(step)
            
        detached_embeds = {'image_features': None,
                           'text_features': None,
                           'logit_scale': None}
        detached_loss_weights = {}
        
        upstream_images, upstream_texts, *upstream_scores, downstream_batches = batch
        for i, (v_images, v_texts, v_cls) in enumerate(downstream_batches):
            downstream_batches[i][0] = v_images.to(device=device, dtype=input_dtype, non_blocking=True)
            downstream_batches[i][1] = v_texts.to(device=device, non_blocking=True)
            downstream_batches[i][2] = torch.Tensor(v_cls).flatten().to(device=device, dtype=torch.int64)

        upstream_images = upstream_images.to(device=device, dtype=input_dtype, non_blocking=True)
        upstream_texts = upstream_texts.to(device=device, non_blocking=True)
        if len(upstream_scores) > 0:
            upstream_scores = upstream_scores[0].to(device=device, non_blocking=True)

        data_time_m.update(time.time() - end)
        
        with torch.no_grad():
            v, l_hat_losses, found_inf, w_loss = compute_v(scoring_params, reference_params)
            l_pre_update = 0
            if args.log_pre_update:
                for downstream_task_name, (downstream_images, downstream_texts, downstream_cls) in zip(downstream_task_names, downstream_batches):
                    l_pre_update += compute_l_pre_update(reference_params, downstream_task_name, downstream_images, downstream_texts, downstream_cls)
            
            if found_inf.item() == 0:
                d_loss_weights = detached_loss_weights['loss_weights']
                dl_dw = grad(compute_loss_to_weights)(d_loss_weights, v, reference_params)
            
                if args.world_size > 1:
                    torch.distributed.all_reduce(dl_dw, op=torch.distributed.ReduceOp.SUM)
                
                if args.train_full_scoring:
                    scoring_params_with_grad = scoring_params
                    scoring_params_without_grad = dict()
                else:
                    scoring_params_with_grad = {k: v for k, v in scoring_params.items() if 'clip_model' not in k}
                    scoring_params_without_grad = {k: v for k, v in scoring_params.items() if 'clip_model' in k}
                
                scaled_scoring_gradients = grad(compute_loss_to_scoring_model)(
                    scoring_params_with_grad, scoring_params_without_grad, dl_dw)
                scoring_gradients, found_inf_scoring = downstream_scaler.unscale_grads(scaled_scoring_gradients)
                
                if args.world_size > 1:
                    for k in scoring_gradients.keys():
                        torch.distributed.all_reduce(scoring_gradients[k], op=torch.distributed.ReduceOp.SUM)
                        scoring_gradients[k] = scoring_gradients[k].detach()

                if found_inf_scoring.item() == 0:
                    scoring_params_with_grad = {k: v.detach() for k, v in scoring_optimizer.get_updated_params(scoring_params_with_grad, scoring_gradients).items()}
                    scoring_params = {**scoring_params_with_grad, **scoring_params_without_grad}
                    
                    if args.update_reference_model:
                        for p in reference_params.keys():
                            reference_params[p] = detached_new_params[p].clone().detach()
                            
                    with torch.no_grad():
                        reference_params['logit_scale'].data.clamp_(2, math.log(100))
                        if 'clip_model.logit_scale' in scoring_params.keys():
                            scoring_params['clip_model.logit_scale'].data.clamp_(2, math.log(100))
                else:
                    logging.info(f"Found {found_inf_scoring.item()} inf or nan in compute_loss_to_scoring_model. Skipping optimizer step.")
                downstream_scaler.update(found_inf_scoring)
            else:
                logging.info(f"Found {found_inf.item()} inf or nan in computing v. Skipping optimizer step.")
        
        sum_downstream_losses = sum(l_hat_losses.values())
        losses = {"l_hat_losses": sum_downstream_losses,
                  "weighted_loss": w_loss,
                  **l_hat_losses}
        
        if args.log_pre_update:
            losses["l_pre_update"] = l_pre_update
            losses["l-l_hat"] = l_pre_update - sum_downstream_losses
        
        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i_accum + 1
        if is_master(args) and (i_accum % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(upstream_images)
            num_samples = batch_count * batch_size * args.accum_freq * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            for key, val in losses.items():
                if key not in losses_m:
                    losses_m[key] = AverageMeter()
                losses_m[key].update(val.item(), batch_size)

            logit_scale_scalar = reference_params['logit_scale'].data.item()
            if args.downstream_logit_scale is not None:
                downstream_logit_scale_scalar = reference_params['downstream_logit_scale'].data.item()
                downstream_logit_scale_str = f"Downstream Logit Scale: {downstream_logit_scale_scalar:.3f} "
            else:
                downstream_logit_scale_str = ""
            
            samples_per_second = args.accum_freq * args.batch_size * args.world_size / batch_time_m.val
            samples_per_second_per_gpu = args.accum_freq * args.batch_size / batch_time_m.val
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
                f"LR: {reference_optimizer.param_groups[0]['lr']:5f} "
                f"L_hat: {losses_m['l_hat_losses'].val:3e} "
                f"Logit Scale: {logit_scale_scalar:.3f} " + downstream_logit_scale_str
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": samples_per_second,
                "samples_per_second_per_gpu": samples_per_second_per_gpu,
                "lr": reference_optimizer.param_groups[0]["lr"],
                "scoring_lr": scoring_optimizer.param_groups[0]["lr"],
                "scale": logit_scale_scalar,
            }
            if args.downstream_logit_scale is not None:
                log_data["downstream_logit_scale"] = downstream_logit_scale_scalar
            
            log_data.update({name:val.val for name,val in losses_m.items()})

            log_data = {"train/" + name: val for name, val in log_data.items()}

            if args.wandb:
                assert wandb is not None, 'Please install wandb.'
                log_data['step'] = step  # for backwards compatibility
                wandb.log(log_data, step=step)
            
            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
            
        scoring_model.load_state_dict(scoring_params)
        reference_model.load_state_dict(reference_params)
    # end for