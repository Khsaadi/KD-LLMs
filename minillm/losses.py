import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from torchtyping import TensorType

from .data_types import PPORLBatch
from .utils import whiten, get_entropy, get_x_entropy, get_log_probs

from transformers import mpu

from utils import all_gather, print_rank

#new 
from knowledge_neurons.knowledge_neurons_mine import KnowledgeNeurons

class Loss():
    def __init__(self, args, trainer):
        self.args = args
        self.trainer = trainer

    def _get_cumsum_rewards(self, rewards):          
        full_rewards = torch.zeros_like(rewards[:, 0])
        for t in reversed(range(rewards.size(1))):
            full_rewards = self.args.gamma * full_rewards + rewards[:, t]
            
        return full_rewards

    def _get_advantages_and_returns(
        self,
        rewards: TensorType["batch_size", "response_size"],
        response_length: int,
        mask: TensorType["batch_size", "response_size"],
        use_whitening: Optional[bool] = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        last_rw = 0
        rw_reversed = []
        
        rewards = rewards.float()
        mask = mask.float()
        lens = torch.cumsum(mask, dim=-1)      # faster way        
        lens = mask - lens + lens[:, -1:None]  # faster way
        lens = torch.masked_fill(lens, lens==0, 1)

        for t in reversed(range(response_length)):
            rw_delta = rewards[:, t]
            last_rw = rw_delta + self.args.gamma * last_rw
            rw_reversed.append(last_rw)

        rw = torch.stack(rw_reversed[::-1], dim=1)
        rw = rw / lens

        advantages = rw

        if use_whitening:
            advantages = whiten(advantages)
        
        return advantages.detach()

    def _pg_loss(
        self,
        logprobs: TensorType["batch_size", "response_size"],
        old_logprobs: TensorType["batch_size", "response_size"],
        advantages: TensorType["batch_size", "response_size"],
        mask: TensorType["batch_size", "response_size"],
        w: TensorType["batch_size", "response_size"],
    ):
        """PPO objective function.
        References:
        - https://stable-baselines.readthedocs.io/en/master/modules/ppo2.html
        """
        n = mask.sum()
        
        log_ratio = (logprobs - old_logprobs) * mask
        ratio = torch.exp(log_ratio.float())            
        ratio = ratio * w

        if any(torch.isinf(advantages).view(-1)):
            print("[ERROR] advantage inf")
        
        if any(torch.isinf(ratio).view(-1)):
            print("[ERROR] ratio inf")

        if any(torch.isnan(advantages).view(-1)):
            print("[ERROR] advantage nan")
        
        if any(torch.isnan(ratio).view(-1)):
            print("[ERROR] ratio nan")
        
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(
            ratio,
            1.0 - self.args.cliprange,
            1.0 + self.args.cliprange,
        )
        pg_loss = torch.sum(torch.max(pg_loss1, pg_loss2).float() * mask) / n

        return pg_loss

    def _reg_loss(self, query_ids, response_ids, mask, logits, inf_mask, stats):
        with torch.no_grad():
            t_logits = self.trainer.compute_logits_and_log_probs(query_ids, response_ids, inf_mask, base="teacher", return_logprobs=False)
        
        loss_exp_ent = 0
        xent = get_x_entropy(logits, t_logits, inf_mask, mask, model_parallel=self.args.model_parallel)
        s_ent = get_entropy(logits, inf_mask, mask, model_parallel=self.args.model_parallel)
        loss_exp_ent = torch.sum((xent - s_ent) * mask) / mask.sum()
        stats["reg_loss"] = loss_exp_ent.item()
        
        return loss_exp_ent

    def get_input_batch(self, ppo_batch: PPORLBatch, pt_batch):
        query_tensors = ppo_batch.query_tensors
        response_tensors = ppo_batch.response_tensors
        ppo_input_batch = self.trainer.get_model_inputs(query_tensors, response_tensors)
        pt_input_batch, _ = pt_batch
        # merge batch
        assert len(ppo_input_batch) == len(pt_input_batch), list(ppo_input_batch.keys())
        input_batch = {}
        for k in ppo_input_batch:
            input_batch[k] = torch.cat([ppo_input_batch[k], pt_input_batch[k]], dim=0)
        return input_batch

    def ppo_loss(self, batch: PPORLBatch, logits):
        stats = {}
        query_tensors = batch.query_tensors
        response_tensors = batch.response_tensors
        lens = batch.lens
        s_lens = batch.s_lens
        mask = batch.mask
        old_logprobs = batch.logprobs
        old_rewards = batch.rewards
        rev_kl = batch.rev_kl
        w = batch.w
        inf_mask = batch.inf_mask
        
        response_length = response_tensors.shape[-1]

        start = query_tensors.size(1) - 1 # "-1" for the first generated token AS TARGET
        end = query_tensors.size(1) + response_tensors.size(1) - 1 # "remove the last token that does not have target"

        logits = logits / self.args.temperature
        logits = logits[:, start:end]
        if inf_mask is not None:
            logits = logits.masked_fill(inf_mask, -float("inf"))
            
        tokens = torch.cat((query_tensors, response_tensors), dim=1)[
            :, -self.trainer.max_length :
        ]
        mask = self.trainer.get_mask(tokens)[:, start:end]
        
        logprobs = get_log_probs(logits, response_tensors, mask, inf_mask, model_parallel=self.args.model_parallel)

        advantages = self._get_advantages_and_returns(
            old_rewards, response_length, mask
        )
        
        loss = self._pg_loss(
            logprobs=logprobs,
            old_logprobs=old_logprobs,
            advantages=advantages,
            mask=mask,
            w=w,
        )
        stats["pg_loss"] = loss.item()
        
        single_step_reg_loss = self._reg_loss(query_tensors, response_tensors, mask, logits, inf_mask, stats)
        stats["reg_loss"] = single_step_reg_loss.item()
        
        if self.args.single_step_reg:
            loss += single_step_reg_loss
        
        stats["rl_loss"] = loss.item()
        
        with torch.no_grad():
            # generation values for reward
            cumsum_rewards = self._get_cumsum_rewards(old_rewards)
            rev_kl = torch.sum(rev_kl, dim=-1)
            
            if self.args.length_norm:
                cumsum_rewards = cumsum_rewards / lens
                rev_kl = rev_kl / s_lens
                        
            cumsum_rewards = all_gather(cumsum_rewards, dim=0, world_size=self.trainer.dp_world_size, group=self.trainer.dp_group).mean(dim=0).item()
            rev_kl = all_gather(rev_kl, dim=0, world_size=self.trainer.dp_world_size, group=self.trainer.dp_group).mean(dim=0).item()
            lens = all_gather(lens, dim=0, world_size=self.trainer.dp_world_size, group=self.trainer.dp_group).float().mean(dim=0).item()
            s_lens = all_gather(s_lens, dim=0, world_size=self.trainer.dp_world_size, group=self.trainer.dp_group).float().mean(dim=0).item()
        
        stats["reward"] = cumsum_rewards
        stats["rev_kl"] = rev_kl
        stats["mixed_lens"] = lens
        stats["stu_lens"] = s_lens
        
        return loss, stats

    def pt_loss(self, batch, logits):
        stats = {}
        model_batch, no_model_batch = batch
        loss_mask = (no_model_batch["label"] != -100).int()
        if self.args.model_parallel:
            lm_losses = mpu.parallel_cross_entropy(logits.contiguous().float(), no_model_batch["label"]).view(-1)
            lm_loss = (lm_losses * loss_mask.view(-1)).sum(-1) / loss_mask.view(-1).sum(-1)
        else:
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            lm_loss = loss_fn(logits.view(-1, logits.size(-1)), no_model_batch["label"].view(-1))
        
        distil_loss = 0
        if self.trainer.teacher_model is not None and self.args.kd_ratio is not None:
            with torch.no_grad():
                teacher_outputs = self.trainer.teacher_model(**model_batch, return_dict=True, use_cache=False)
                teacher_logits = teacher_outputs.logits
            if self.args.model_parallel:
                distil_losses = mpu.parallel_soft_cross_entropy_loss(logits.float(), teacher_logits.float())
                distil_losses = distil_losses.view(-1)
                distil_loss = (distil_losses * loss_mask.view(-1)).sum(-1) / loss_mask.view(-1).sum(-1)
            else:
                teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
                inf_mask = torch.isinf(logits)
                logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
                prod_probs = torch.masked_fill(teacher_probs * logprobs, inf_mask, 0)
                x = torch.sum(prod_probs, dim=-1).view(-1)
                distil_loss = -torch.sum(x * loss_mask.view(-1), dim=0) / torch.sum(loss_mask.view(-1), dim=0)
            
            loss = (1-self.args.kd_ratio) * lm_loss + self.args.kd_ratio * distil_loss

        stats["pt_loss"] = loss.item()
        stats["lm_loss"] = lm_loss.item()
        stats["ds_loss"] = distil_loss.item()

        return loss, stats
    
    def ft_loss(self,batch):
        # neurons = get_refined_neurons()
        stats = {}
        model_batch, no_model_batch = batch
        # loss_mask = (no_model_batch["label"] != -100).int()
        # ft_distil_loss = 0
        if self.trainer.teacher_model is not None and self.args.ft_ratio is not None:
            with torch.no_grad():
                KN = KnowledgeNeurons(self.trainer.teacher_model, self.tokenizer)
                scores = KN.get_scores(model_batch, 16, 20, "integrated_grads", True)
                teacher_outputs = self.trainer.teacher_model(**model_batch, output_hidden_states=True, return_dict=True, use_cache=False)
                

            K = 3072  # Choose how many top hidden units you want
            top_k_indices = torch.topk(scores, K).indices
            important_features = teacher_outputs.hidden_states[-1][0, :, top_k_indices[0]]
            student_last_layer = self.forward_model(**model_batch, output_hidden_states=True, return_dict=True, use_cache=False).hidden_states[-1]
            feature_loss = F.mse_loss(important_features, student_last_layer)

            # if self.args.model_parallel:
            #     distil_losses = mpu.parallel_soft_cross_entropy_loss(logits.float(), teacher_logits.float())
            #     distil_losses = distil_losses.view(-1)
            #     distil_loss = (distil_losses * loss_mask.view(-1)).sum(-1) / loss_mask.view(-1).sum(-1)
            # else:
            #     teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
            #     inf_mask = torch.isinf(logits)
            #     logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
            #     prod_probs = torch.masked_fill(teacher_probs * logprobs, inf_mask, 0)
            #     x = torch.sum(prod_probs, dim=-1).view(-1)
            #     distil_loss = -torch.sum(x * loss_mask.view(-1), dim=0) / torch.sum(loss_mask.view(-1), dim=0)
            
            loss = self.args.ft_ratio * feature_loss
        # model_batch, no_model_batch = batch
        # loss_mask = (no_model_batch["label"] != -100).int()
        # if self.args.model_parallel:
        #     lm_losses = mpu.parallel_cross_entropy(logits.contiguous().float(), no_model_batch["label"]).view(-1)
        #     lm_loss = (lm_losses * loss_mask.view(-1)).sum(-1) / loss_mask.view(-1).sum(-1)
        # else:
        #     loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        #     lm_loss = loss_fn(logits.view(-1, logits.size(-1)), no_model_batch["label"].view(-1))
        
        # distil_loss = 0
        # if self.trainer.teacher_model is not None and self.args.kd_ratio is not None:
        #     with torch.no_grad():
        #         teacher_outputs = self.trainer.teacher_model(**model_batch, return_dict=True, use_cache=False)
        #         teacher_logits = teacher_outputs.logits
        #     if self.args.model_parallel:
        #         distil_losses = mpu.parallel_soft_cross_entropy_loss(logits.float(), teacher_logits.float())
        #         distil_losses = distil_losses.view(-1)
        #         distil_loss = (distil_losses * loss_mask.view(-1)).sum(-1) / loss_mask.view(-1).sum(-1)
        #     else:
        #         teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
        #         inf_mask = torch.isinf(logits)
        #         logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
        #         prod_probs = torch.masked_fill(teacher_probs * logprobs, inf_mask, 0)
        #         x = torch.sum(prod_probs, dim=-1).view(-1)
        #         distil_loss = -torch.sum(x * loss_mask.view(-1), dim=0) / torch.sum(loss_mask.view(-1), dim=0)
            
        #     loss = (1-self.args.kd_ratio) * lm_loss + self.args.kd_ratio * distil_loss

        stats["ft_loss"] = loss.item()

        return loss, stats



    #     coarse_neurons.append(
    #     self.get_coarse_neurons(
    #     prompt,
    #     ground_truth,
    #     batch_size=batch_size,
    #     steps=steps,
    #     adaptive_threshold=coarse_adaptive_threshold,
    #     threshold=coarse_threshold,
    #     percentile=coarse_percentile,
    #     pbar=False,
    # ))

    #     return loss, stats
    

    # def get_coarse_neurons(
    #     self,
    #     prompt: str,
    #     ground_truth: str,
    #     batch_size: int = 10,
    #     steps: int = 20,
    #     threshold: float = None,
    #     adaptive_threshold: float = None,
    #     percentile: float = None,
    #     attribution_method: str = "integrated_grads",
    #     pbar: bool = True,
    # ) -> List[List[int]]:
    #     """
    #     Finds the 'coarse' neurons for a given prompt and ground truth.
    #     The coarse neurons are the neurons that are most activated by a single prompt.
    #     We refine these by using multiple prompts that express the same 'fact'/relation in different ways.

    #     `prompt`: str
    #         the prompt to get the coarse neurons for
    #     `ground_truth`: str
    #         the ground truth / expected output
    #     `batch_size`: int
    #         batch size
    #     `steps`: int
    #         total number of steps (per token) for the integrated gradient calculations
    #     `threshold`: float
    #         `t` from the paper. If not None, then we only keep neurons with integrated grads above this threshold.
    #     `adaptive_threshold`: float
    #         Adaptively set `threshold` based on `maximum attribution score * adaptive_threshold` (in the paper, they set adaptive_threshold=0.3)
    #     `percentile`: float
    #         If not None, then we only keep neurons with integrated grads in this percentile of all integrated grads.
    #     `attribution_method`: str
    #         the method to use for getting the scores. Choose from 'integrated_grads' or 'max_activations'.
    #     """
    #     attribution_scores = self.get_scores(
    #         prompt,
    #         ground_truth,
    #         batch_size=batch_size,
    #         steps=steps,
    #         pbar=pbar,
    #         attribution_method=attribution_method,
    #     )
    #     assert (
    #         sum(e is not None for e in [threshold, adaptive_threshold, percentile]) == 1
    #     ), f"Provide one and only one of threshold / adaptive_threshold / percentile"
    #     if adaptive_threshold is not None:
    #         threshold = attribution_scores.max().item() * adaptive_threshold
    #     if threshold is not None:
    #         return torch.nonzero(attribution_scores > threshold).cpu().tolist()
    #     else:
    #         s = attribution_scores.flatten().detach().cpu().numpy()
    #         return (
    #             torch.nonzero(attribution_scores > np.percentile(s, percentile))
    #             .cpu()
    #             .tolist()
    #         )
    
    # def get_scores(
    #     self,
    #     prompt: str,
    #     ground_truth: str,
    #     batch_size: int = 10,
    #     steps: int = 20,
    #     attribution_method: str = "integrated_grads",
    #     pbar: bool = True,
    # ):
    #     """
    #     Gets the attribution scores for a given prompt and ground truth.
    #     `prompt`: str
    #         the prompt to get the attribution scores for
    #     `ground_truth`: str
    #         the ground truth / expected output
    #     `batch_size`: int
    #         batch size
    #     `steps`: int
    #         total number of steps (per token) for the integrated gradient calculations
    #     `attribution_method`: str
    #         the method to use for getting the scores. Choose from 'integrated_grads' or 'max_activations'.
    #     """

    #     scores = []
    #     encoded_input = self.tokenizer(prompt, return_tensors="pt").to(self.device)
    #     for layer_idx in tqdm(
    #         range(self.n_layers()),
    #         desc="Getting attribution scores for each layer...",
    #         disable=not pbar,
    #     ):
    #         layer_scores = self.get_scores_for_layer(
    #             prompt,
    #             ground_truth,
    #             encoded_input=encoded_input,
    #             layer_idx=layer_idx,
    #             batch_size=batch_size,
    #             steps=steps,
    #             attribution_method=attribution_method,
    #         )
    #         scores.append(layer_scores)
    #     return torch.stack(scores)
       
    # def get_scores_for_layer(
    #     self,
    #     prompt: str,
    #     ground_truth: str,
    #     layer_idx: int,
    #     batch_size: int = 10,
    #     steps: int = 20,
    #     encoded_input: Optional[int] = None,
    #     attribution_method: str = "integrated_grads",
    # ):
    #     """
    #     get the attribution scores for a given layer
    #     `prompt`: str
    #         the prompt to get the attribution scores for
    #     `ground_truth`: str
    #         the ground truth / expected output
    #     `layer_idx`: int
    #         the layer to get the scores for
    #     `batch_size`: int
    #         batch size
    #     `steps`: int
    #         total number of steps (per token) for the integrated gradient calculations
    #     `encoded_input`: int
    #         if not None, then use this encoded input instead of getting a new one
    #     `attribution_method`: str
    #         the method to use for getting the scores. Choose from 'integrated_grads' or 'max_activations'.
    #     """
    #     assert steps % batch_size == 0
    #     n_batches = steps // batch_size

    #     # First we take the unmodified model and use a hook to return the baseline intermediate activations at our chosen target layer
    #     encoded_input, mask_idx, target_label = self._prepare_inputs(
    #         prompt, ground_truth, encoded_input
    #     )

    #     # for autoregressive models, we might want to generate > 1 token
    #     if self.model_type == "gpt":
    #         n_sampling_steps = len(target_label)
    #     else:
    #         n_sampling_steps = 1  # TODO: we might want to use multiple mask tokens even with bert models

    #     if attribution_method == "integrated_grads":
    #         integrated_grads = []

    #         for i in range(n_sampling_steps):
    #             if i > 0 and self.model_type == "gpt":
    #                 # retokenize new inputs
    #                 encoded_input, mask_idx, target_label = self._prepare_inputs(
    #                     prompt, ground_truth
    #                 )
    #             (
    #                 baseline_outputs,
    #                 baseline_activations,
    #             ) = self.get_baseline_with_activations(
    #                 encoded_input, layer_idx, mask_idx
    #             )
    #             if n_sampling_steps > 1:
    #                 argmax_next_token = (
    #                     baseline_outputs.logits[:, mask_idx, :].argmax(dim=-1).item()
    #                 )
    #                 next_token_str = self.tokenizer.decode(argmax_next_token)

    #             # Now we want to gradually change the intermediate activations of our layer from 0 -> their original value
    #             # and calculate the integrated gradient of the masked position at each step
    #             # we do this by repeating the input across the batch dimension, multiplying the first batch by 0, the second by 0.1, etc., until we reach 1
    #             scaled_weights = self.scaled_input(
    #                 baseline_activations, steps=steps, device=self.device
    #             )
    #             scaled_weights.requires_grad_(True)

    #             integrated_grads_this_step = []  # to store the integrated gradients

    #             for batch_weights in scaled_weights.chunk(n_batches):
    #                 # we want to replace the intermediate activations at some layer, at the mask position, with `batch_weights`
    #                 # first tile the inputs to the correct batch size
    #                 inputs = {
    #                     "input_ids": einops.repeat(
    #                         encoded_input["input_ids"], "b d -> (r b) d", r=batch_size
    #                     ),
    #                     "attention_mask": einops.repeat(
    #                         encoded_input["attention_mask"],
    #                         "b d -> (r b) d",
    #                         r=batch_size,
    #                     ),
    #                 }
    #                 if self.model_type == "bert":
    #                     inputs["token_type_ids"] = einops.repeat(
    #                         encoded_input["token_type_ids"],
    #                         "b d -> (r b) d",
    #                         r=batch_size,
    #                     )

    #                 # then patch the model to replace the activations with the scaled activations
    #                 patch_ff_layer(
    #                     self.model,
    #                     layer_idx=layer_idx,
    #                     mask_idx=mask_idx,
    #                     replacement_activations=batch_weights,
    #                     transformer_layers_attr=self.transformer_layers_attr,
    #                     ff_attrs=self.input_ff_attr,
    #                 )

    #                 # then forward through the model to get the logits
    #                 outputs = self.model(**inputs)

    #                 # then calculate the gradients for each step w/r/t the inputs
    #                 probs = F.softmax(outputs.logits[:, mask_idx, :], dim=-1)
    #                 if n_sampling_steps > 1:
    #                     target_idx = target_label[i]
    #                 else:
    #                     target_idx = target_label
    #                 grad = torch.autograd.grad(
    #                     torch.unbind(probs[:, target_idx]), batch_weights
    #                 )[0]
    #                 grad = grad.sum(dim=0)
    #                 integrated_grads_this_step.append(grad)

    #                 unpatch_ff_layer(
    #                     self.model,
    #                     layer_idx=layer_idx,
    #                     transformer_layers_attr=self.transformer_layers_attr,
    #                     ff_attrs=self.input_ff_attr,
    #                 )

    #             # then sum, and multiply by W-hat / m
    #             integrated_grads_this_step = torch.stack(
    #                 integrated_grads_this_step, dim=0
    #             ).sum(dim=0)
    #             integrated_grads_this_step *= baseline_activations.squeeze(0) / steps
    #             integrated_grads.append(integrated_grads_this_step)

    #             if n_sampling_steps > 1:
    #                 prompt += next_token_str
    #         integrated_grads = torch.stack(integrated_grads, dim=0).sum(dim=0) / len(
    #             integrated_grads
    #         )
    #         return integrated_grads
    #     elif attribution_method == "max_activations":
    #         activations = []
    #         for i in range(n_sampling_steps):
    #             if i > 0 and self.model_type == "gpt":
    #                 # retokenize new inputs
    #                 encoded_input, mask_idx, target_label = self._prepare_inputs(
    #                     prompt, ground_truth
    #                 )
    #             (
    #                 baseline_outputs,
    #                 baseline_activations,
    #             ) = self.get_baseline_with_activations(
    #                 encoded_input, layer_idx, mask_idx
    #             )
    #             activations.append(baseline_activations)
    #             if n_sampling_steps > 1:
    #                 argmax_next_token = (
    #                     baseline_outputs.logits[:, mask_idx, :].argmax(dim=-1).item()
    #                 )
    #                 next_token_str = self.tokenizer.decode(argmax_next_token)
    #                 prompt += next_token_str
    #         activations = torch.stack(activations, dim=0).sum(dim=0) / len(activations)
    #         return activations.squeeze(0)
    #     else:
    #         raise NotImplementedError
        

    # def _prepare_inputs(self, prompt, target=None, encoded_input=None):
    #     if encoded_input is None:
    #         encoded_input = self.tokenizer(prompt, return_tensors="pt").to(self.device)
    #     if self.model_type == "bert":
    #         mask_idx = torch.where(
    #             encoded_input["input_ids"][0] == self.tokenizer.mask_token_id
    #         )[0].item()
    #     else:
    #         # with autoregressive models we always want to target the last token
    #         mask_idx = -1
    #     if target is not None:
    #         if "gpt" in self.model_type:
    #             target = self.tokenizer.encode(target)
    #         else:
    #             target = self.tokenizer.convert_tokens_to_ids(target)
    #     return encoded_input, mask_idx, target
    
    # def get_baseline_with_activations(
    #     self, encoded_input: dict, layer_idx: int, mask_idx: int
    # ):
    #     """
    #     Gets the baseline outputs and activations for the unmodified model at a given index.

    #     `encoded_input`: torch.Tensor
    #         the inputs to the model from self.tokenizer.encode_plus()
    #     `layer_idx`: int
    #         which transformer layer to access
    #     `mask_idx`: int
    #         the position at which to get the activations (TODO: rename? with autoregressive models there's no mask, so)
    #     """

    #     def get_activations(model, layer_idx, mask_idx):
    #         """
    #         This hook function should assign the intermediate activations at a given layer / mask idx
    #         to the 'self.baseline_activations' variable
    #         """

    #         def hook_fn(acts):
    #             self.baseline_activations = acts[:, mask_idx, :]

    #         return register_hook(
    #             model,
    #             layer_idx=layer_idx,
    #             f=hook_fn,
    #             transformer_layers_attr=self.transformer_layers_attr,
    #             ff_attrs=self.input_ff_attr,
    #         )

    #     handle = get_activations(self.model, layer_idx=layer_idx, mask_idx=mask_idx)
    #     baseline_outputs = self.model(**encoded_input)
    #     handle.remove()
    #     baseline_activations = self.baseline_activations
    #     self.baseline_activations = None
    #     return baseline_outputs, baseline_activations
    

    # @staticmethod
    # def scaled_input(activations: torch.Tensor, steps: int = 20, device: str = "cpu"):
    #     """
    #     Tiles activations along the batch dimension - gradually scaling them over
    #     `steps` steps from 0 to their original value over the batch dimensions.

    #     `activations`: torch.Tensor
    #     original activations
    #     `steps`: int
    #     number of steps to take
    #     """
    #     tiled_activations = einops.repeat(activations, "b d -> (r b) d", r=steps)
    #     out = (
    #         tiled_activations
    #         * torch.linspace(start=0, end=1, steps=steps).to(device)[:, None]
    #     )
    #     return out
    

    # class Patch(torch.nn.Module):
    #     """
    #     Patches a torch module to replace/suppress/enhance the intermediate activations
    #     """

    #     def __init__(
    #         self,
    #         ff_layer: nn.Module,
    #         mask_idx: int,
    #         replacement_activations: torch.Tensor = None,
    #         target_positions: List[List[int]] = None,
    #         mode: str = "replace",
    #         enhance_value: float = 2.0,
    #     ):
    #         super().__init__()
    #         self.ff = ff_layer
    #         self.acts = replacement_activations
    #         self.mask_idx = mask_idx
    #         self.target_positions = target_positions
    #         self.enhance_value = enhance_value
    #         assert mode in ["replace", "suppress", "enhance"]
    #         self.mode = mode
    #         if self.mode == "replace":
    #             assert self.acts is not None
    #         elif self.mode in ["enhance", "suppress"]:
    #             assert self.target_positions is not None

    #     def forward(self, x: torch.Tensor):
    #         x = self.ff(x)
    #         if self.mode == "replace":
    #             x[:, self.mask_idx, :] = self.acts
    #         elif self.mode == "suppress":
    #             for pos in self.target_positions:
    #                 x[:, self.mask_idx, pos] = 0.0
    #         elif self.mode == "enhance":
    #             for pos in self.target_positions:
    #                 x[:, self.mask_idx, pos] *= self.enhance_value
    #         else:
    #             raise NotImplementedError
    #         return x
    # def patch_ff_layer(
    #     model: nn.Module,
    #     mask_idx: int,
    #     layer_idx: int = None,
    #     replacement_activations: torch.Tensor = None,
    #     mode: str = "replace",
    #     transformer_layers_attr: str = "bert.encoder.layer",
    #     ff_attrs: str = "intermediate",
    #     neurons: List[List[int]] = None,
    # ):
    #     """
    #     replaces the ff layer at `layer_idx` with a `Patch` class - that will replace the intermediate activations at sequence position
    #     `mask_index` with `replacement_activations`

    #     `model`: nn.Module
    #     a torch.nn.Module [currently only works with HF Bert models]
    #     `layer_idx`: int
    #     which transformer layer to access
    #     `mask_idx`: int
    #     the index (along the sequence length) of the activation to replace.
    #     TODO: multiple indices
    #     `replacement_activations`: torch.Tensor
    #     activations [taken from the mask_idx position of the unmodified activations] of shape [b, d]
    #     `transformer_layers_attr`: str
    #     chain of attributes (separated by periods) that access the transformer layers within `model`.
    #     The transformer layers are expected to be indexable - i.e a Modulelist
    #     `ff_attrs`: str
    #     chain of attributes (separated by periods) that access the ff block within a transformer layer
    #     """
    #     transformer_layers = get_attributes(model, transformer_layers_attr)

    #     if mode == "replace":
    #         ff_layer = get_attributes(transformer_layers[layer_idx], ff_attrs)
    #         assert layer_idx < len(
    #             transformer_layers
    #         ), f"cannot get layer {layer_idx + 1} of a {len(transformer_layers)} layer model"

    #         set_attribute_recursive(
    #             transformer_layers[layer_idx],
    #             ff_attrs,
    #             Patch(
    #                 ff_layer,
    #                 mask_idx,
    #                 replacement_activations=replacement_activations,
    #                 mode=mode,
    #             ),
    #         )

    #     elif mode in ["suppress", "enhance"]:
    #         neurons_dict = collections.defaultdict(list)
    #         for neuron in neurons:
    #             layer_idx, pos = neuron
    #             neurons_dict[layer_idx].append(pos)
    #         for layer_idx, positions in neurons_dict.items():
    #             assert layer_idx < len(transformer_layers)
    #             ff_layer = get_attributes(transformer_layers[layer_idx], ff_attrs)
    #             set_attribute_recursive(
    #                 transformer_layers[layer_idx],
    #                 ff_attrs,
    #                 Patch(
    #                     ff_layer,
    #                     mask_idx,
    #                     replacement_activations=None,
    #                     mode=mode,
    #                     target_positions=positions,
    #                 ),
    #             )
    #     else:
    #         raise NotImplementedError
        
    # def unpatch_ff_layer(
    #     model: nn.Module,
    #     layer_idx: int,
    #     transformer_layers_attr: str = "bert.encoder.layer",
    #     ff_attrs: str = "intermediate",
    # ):
    #     """
    #     Removes the `Patch` applied by `patch_ff_layer`, replacing it with its original value.

    #     `model`: torch.nn.Module
    #     a torch.nn.Module [currently only works with HF Bert models]
    #     `layer_idx`: int
    #     which transformer layer to access
    #     `transformer_layers_attr`: str
    #     chain of attributes (separated by periods) that access the transformer layers within `model`.
    #     The transformer layers are expected to be indexable - i.e a Modulelist
    #     `ff_attrs`: str
    #     chain of attributes (separated by periods) that access the ff block within a transformer layer
    #     """
    #     transformer_layers = get_attributes(model, transformer_layers_attr)
    #     assert layer_idx < len(
    #         transformer_layers
    #     ), f"cannot get layer {layer_idx + 1} of a {len(transformer_layers)} layer model"
    #     ff_layer = get_attributes(transformer_layers[layer_idx], ff_attrs)
    #     assert isinstance(ff_layer, Patch), "Can't unpatch a layer that hasn't been patched"
    #     set_attribute_recursive(
    #         transformer_layers[layer_idx],
    #         ff_attrs,
    #         ff_layer.ff,
    #     )


    # def unpatch_ff_layers(
    #     model: nn.Module,
    #     layer_indices: int,
    #     transformer_layers_attr: str = "bert.encoder.layer",
    #     ff_attrs: str = "intermediate",
    # ):
    #     """
    #     Calls unpatch_ff_layer for all layers in layer_indices
    #     """
    #     for layer_idx in layer_indices:
    #         unpatch_ff_layer(model, layer_idx, transformer_layers_attr, ff_attrs)