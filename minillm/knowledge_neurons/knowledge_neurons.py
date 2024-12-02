# main knowledge neurons class
import torch
import torch.nn.functional as F
import torch.nn as nn
import einops
from tqdm import tqdm
import numpy as np
import collections
from typing import List, Optional, Tuple, Callable
import torch
import torch.nn.functional as F
import einops
import collections
import math
from functools import partial
from transformers import PreTrainedTokenizerBase
from .patch import *
from ..utils import  get_log_probs



class KnowledgeNeurons:
    def __init__(
        self,
        max_length,
        args,
        model: nn.Module,
        device: str = None,
    ):
        self.args = args
        self.model = model
        self.max_length = max_length
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        # self.tokenizer = tokenizer

        self.baseline_activations = None
        self.transformer_layers_attr = "transformer.h"
        self.input_ff_attr = "mlp.c_fc"
        self.output_ff_attr = "mlp.c_proj.weight"
        self.word_embeddings_attr = "transformer.wpe"

    def _get_output_ff_layer(self, layer_idx):
        return get_ff_layer(
            self.model,
            layer_idx,
            transformer_layers_attr=self.transformer_layers_attr,
            ff_attrs=self.output_ff_attr,
        )

    def _get_input_ff_layer(self, layer_idx):
        return get_ff_layer(
            self.model,
            layer_idx,
            transformer_layers_attr=self.transformer_layers_attr,
            ff_attrs=self.input_ff_attr,
        )

    def _get_word_embeddings(self):
        return get_attributes(self.model, self.word_embeddings_attr)

    def _get_transformer_layers(self):
        return get_attributes(self.model, self.transformer_layers_attr)



    def n_layers(self):
        return len(self._get_transformer_layers())

    def intermediate_size(self):
            return self.model.config.hidden_size * 4

    @staticmethod
    def scaled_input(activations: torch.Tensor, steps: int = 20, device: str = "cpu"):
        """
        Tiles activations along the batch dimension - gradually scaling them over
        `steps` steps from 0 to their original value over the batch dimensions.

        `activations`: torch.Tensor
        original activations
        `steps`: int
        number of steps to take
        """
        tiled_activations = einops.repeat(activations, "b d -> (r b) d ", r=steps)
        scaling = torch.linspace(start=0, end=1, steps=steps).to(device)
        scaling = scaling.repeat_interleave(int(tiled_activations.size(0)/steps))[:, None]
        out = tiled_activations * scaling
        # out = (
        #     tiled_activations
        #     * torch.linspace(start=0, end=1, steps=steps).to(device)[:, None]
        # )
        return out
    
    def get_baseline_with_activations_1(self, encoded_input: dict, layer_idx: int, start: int, end: int):
        """
        Gets the baseline outputs and activations for the unmodified model at a given index.

        `encoded_input`: torch.Tensor
            the inputs to the model from self.tokenizer.encode_plus()
        `layer_idx`: int
            which transformer layer to access
        `mask_idx`: int
            the position at which to get the activations (TODO: rename? with autoregressive models there's no mask, so)
        """

        def get_activations(model, layer_idx, start, end):
            """
            This hook function should assign the intermediate activations at a given layer / mask idx
            to the 'self.baseline_activations' variable
            """

            def hook_fn(acts):
                self.baseline_activations = acts[:, start:end, :]

            return register_hook(
                model,
                layer_idx=layer_idx,
                f=hook_fn,
                transformer_layers_attr=self.transformer_layers_attr,
                ff_attrs=self.input_ff_attr,
            )

        handle = get_activations(self.model, layer_idx=layer_idx, start=start, end=end)
        with torch.no_grad():
             baseline_outputs = self.model(**encoded_input,output_hidden_states=True)
        #grad = torch.autograd.grad(baseline_outputs.logits, baseline_outputs.hidden_states[-1],grad_outputs=torch.ones_like(baseline_outputs.logits), retain_graph=True)[0]
        handle.remove()
        baseline_activations = self.baseline_activations
        self.baseline_activations = None
        return baseline_outputs, baseline_activations
    
    def get_baseline_with_activations(self, encoded_input: dict, layer_idx: int, start: int):
        """
        Gets the baseline outputs and activations for the unmodified model at a given index.

        `encoded_input`: torch.Tensor
            the inputs to the model from self.tokenizer.encode_plus()
        `layer_idx`: int
            which transformer layer to access
        `mask_idx`: int
            the position at which to get the activations (TODO: rename? with autoregressive models there's no mask, so)
        """

        def get_activations(model, layer_idx, start):
            """
            This hook function should assign the intermediate activations at a given layer / mask idx
            to the 'self.baseline_activations' variable
            """

            def hook_fn(acts):
                self.baseline_activations = acts[:, start, :]

            return register_hook(
                model,
                layer_idx=layer_idx,
                f=hook_fn,
                transformer_layers_attr=self.transformer_layers_attr,
                ff_attrs=self.input_ff_attr,
            )

        handle = get_activations(self.model, layer_idx=layer_idx, start=start)
        baseline_outputs = self.model(**encoded_input,output_hidden_states=True)
        #grad = torch.autograd.grad(baseline_outputs.logits, baseline_outputs.hidden_states[-1],grad_outputs=torch.ones_like(baseline_outputs.logits), retain_graph=True)[0]
        handle.remove()
        baseline_activations = self.baseline_activations
        self.baseline_activations = None
        return baseline_outputs, baseline_activations

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

    def get_scores(
        self,
        tokenizer,
        batch,
        query_ids, 
        inf_mask, 
        response_ids,
        # encoded_input,
        batch_size: int = 16,
        # steps: int = 20,
        steps: int = 16,
        attribution_method: str = "integrated_grads",
        pbar: bool = True,
    ):
        """
        Gets the attribution scores for a given prompt and ground truth.
        `prompt`: str
            the prompt to get the attribution scores for
        `ground_truth`: str
            the ground truth / expected output
        `batch_size`: int
            batch size
        `steps`: int
            total number of steps (per token) for the integrated gradient calculations
        `attribution_method`: str
            the method to use for getting the scores. Choose from 'integrated_grads' or 'max_activations'.
        """

        mlp_scores = []
        hidd_scores = []
        scores = []
        # for layer_idx in tqdm(
        #     range(self.n_layers()),
        #     desc="Getting attribution scores for each layer...",
        #     disable=not pbar,
        # ):
        import ast
        teacher_layer_indices = ast.literal_eval(self.args.teacher_layer_indices)
        baseline_outputs = self.model(**batch,output_hidden_states=True)
        if (self.args.mlp == True and self.args.hidden == True):
            for layer_idx in teacher_layer_indices:
                hidden_layer_scores , mlp_layer_scores= self.get_scores_for_layer(
                    # batch,
                    baseline_outputs,
                    tokenizer,
                    query_ids, 
                    inf_mask, 
                    response_ids,
                    layer_idx=layer_idx,
                    batch_size=batch_size,
                    steps=steps,
                    encoded_input=batch,

                    attribution_method=attribution_method,
                )
                mlp_scores.append(mlp_layer_scores)
                hidd_scores.append(hidden_layer_scores)
            #return torch.stack(mlp_scores), torch.stack(hidd_scores)
            return mlp_scores, hidd_scores
        else:
            for layer_idx in teacher_layer_indices:
                layer_scores = self.get_scores_for_layer(
                    # batch,
                    baseline_outputs,
                    tokenizer,
                    query_ids, 
                    inf_mask, 
                    response_ids,
                    layer_idx=layer_idx,
                    batch_size=batch_size,
                    steps=steps,
                    encoded_input=batch,

                    attribution_method=attribution_method,
                )
                scores.append(layer_scores)
        
            #return torch.stack(scores)
            return scores

    # def get_scores(
    #     self,
    #     tokenizer,
    #     batch,
    #     query_ids, 
    #     inf_mask, 
    #     response_ids,
    #     # encoded_input,
    #     batch_size: int = 16,
    #     # steps: int = 20,
    #     steps: int = 16,
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

    #     mlp_scores = []
    #     hidd_scores = []
    #     scores = []
    #     # for layer_idx in tqdm(
    #     #     range(self.n_layers()),
    #     #     desc="Getting attribution scores for each layer...",
    #     #     disable=not pbar,
    #     # ):
    #     if (self.args.mlp == True and self.args.hidden == True):
    #         for layer_idx in [self.args.layer_index]:
    #             hidden_layer_scores , mlp_layer_scores= self.get_scores_for_layer(
    #                 # batch,
    #                 tokenizer,
    #                 query_ids, 
    #                 inf_mask, 
    #                 response_ids,
    #                 layer_idx=layer_idx,
    #                 batch_size=batch_size,
    #                 steps=steps,
    #                 encoded_input=batch,

    #                 attribution_method=attribution_method,
    #             )
    #             mlp_scores.append(mlp_layer_scores)
    #             hidd_scores.append(hidden_layer_scores)
    #         return torch.stack(mlp_scores), torch.stack(hidd_scores)
    #     else:
    #         for layer_idx in [self.args.layer_index]:
    #             layer_scores = self.get_scores_for_layer(
    #                 # batch,
    #                 tokenizer,
    #                 query_ids, 
    #                 inf_mask, 
    #                 response_ids,
    #                 layer_idx=layer_idx,
    #                 batch_size=batch_size,
    #                 steps=steps,
    #                 encoded_input=batch,

    #                 attribution_method=attribution_method,
    #             )
    #             scores.append(layer_scores)
        
    #         return torch.stack(scores)


    def _prepare_inputs(self, prompt, target=None, encoded_input=None):
        if encoded_input is None:
            encoded_input = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        else:
            # with autoregressive models we always want to target the last token
            mask_idx = -1
        if target is not None:
            target = self.tokenizer.encode(target)
        return encoded_input, mask_idx, target
    
    def get_mask(self, tokenizer, tokens):
        attention_mask = (
            tokens.not_equal(tokenizer.pad_token_id).long()
        )
        return attention_mask
    
    def get_model_inputs(
        self,
        tokenizer,
        query_tensors,
        response_tensors,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens = torch.cat((query_tensors, response_tensors), dim=1)[
            :, -self.max_length :
        ]
        attention_mask = self.get_mask(tokenizer,tokens)
  
        batch = {
            "input_ids": tokens,
            "attention_mask": attention_mask
        }
        
        # if self.args.model_type in ["gpt2"]:  
            # For a proper positional encoding in case of left padding
        position_ids = attention_mask.cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask.eq(0), 0)
        batch["position_ids"] = position_ids
        
        return batch
    
    def get_scores_for_layer(
        self,
        # batch,
        baseline_outputs,
        tokenizer,
        query_ids, inf_mask, response_ids,
        layer_idx: int,
        batch_size: int = 2,
        # steps: int = 20,
        steps: int = 16,
        encoded_input: Optional[int] = None,
        
        attribution_method: str = "integrated_grads",
    ):
        """
        get the attribution scores for a given layer
        `prompt`: str
            the prompt to get the attribution scores for
        `ground_truth`: str
            the ground truth / expected output
        `layer_idx`: int
            the layer to get the scores for
        `batch_size`: int
            batch size
        `steps`: int
            total number of steps (per token) for the integrated gradient calculations
        `encoded_input`: int
            if not None, then use this encoded input instead of getting a new one
        `attribution_method`: str
            the method to use for getting the scores. Choose from 'integrated_grads' or 'max_activations'.
        """
        assert steps % batch_size == 0
        n_batches = steps // batch_size
        response_tensors = response_ids
        query_tensors = query_ids
        response_tensors = einops.repeat(response_tensors, "b d -> (r b) d ", r=batch_size)
        inf_mask= einops.repeat(inf_mask, "b d c -> (r b) d c", r=batch_size)
        if attribution_method == "integrated_grads":
            start = query_tensors.size(1) - 1
            end = query_tensors.size(1) + response_tensors.size(1) - 1
            mask_idx = end-1
            (  baseline_outputs,
                baseline_activations,
            ) = self.get_baseline_with_activations(
                encoded_input, layer_idx, mask_idx
            )
            scaled_weights = self.scaled_input(
                baseline_activations, steps=steps, device=self.device
            )
            scaled_weights.requires_grad_(True)

            integrated_grads_this_step = []  # to store the integrated gradients
            #n_batches = 4
            all_gradients = []
            for batch_weights in scaled_weights.chunk(n_batches):
            
                inputs = {
                    "input_ids": einops.repeat(
                        encoded_input["input_ids"], "b d -> (r b) d", r=batch_size
                    ),
                    "attention_mask": einops.repeat(
                        encoded_input["attention_mask"],
                        "b d -> (r b) d",
                        r=batch_size,
                    ),
                }
                
                indices = response_tensors.size(1) - 1
                patch_ff_layer(
                    self.model,
                    mask_idx=mask_idx,
                    # start = start,
                    # end = end,
                    layer_idx=layer_idx,
                    
                    replacement_activations=batch_weights,
                    transformer_layers_attr=self.transformer_layers_attr,
                    ff_attrs=self.input_ff_attr,
                )
                outputs = self.model(
                **inputs,
                return_dict=True,
                output_hidden_states=True,
                use_cache=False
                )

                # then calculate the gradients for each step w/r/t the inputs
                logits = outputs.logits
                logits = logits[:, mask_idx, :]
                probs = F.softmax(logits, dim=-1)
                #prob = probs[:,response_tensors[:,0]]
                prob = torch.gather(probs, dim=-1, index=response_tensors[:,end-1-start].unsqueeze(-1)).squeeze(-1)
                grad = torch.autograd.grad(torch.unbind(prob), batch_weights
                    )[0]
                grad = grad * baseline_activations.squeeze(0) / steps
                integrated_grads_this_step.append(grad.sum(dim=0))


                unpatch_ff_layer(
                    self.model,
                    layer_idx=layer_idx,
                    transformer_layers_attr=self.transformer_layers_attr,
                    ff_attrs=self.input_ff_attr,
                )

            integrated_grads_this_step = torch.stack(integrated_grads_this_step, dim=0).sum(dim=0) /steps
            return integrated_grads_this_step # add hidden_state
    
        elif attribution_method == "max_activations":
            # scores_hidd, scores_mlp = None, None
            start = query_tensors.size(1) - 1
            end = query_tensors.size(1) + response_tensors.size(1) - 1
            output, baseline_activations_mlp = self.get_baseline_with_activations_1(encoded_input, layer_idx, start, end)
            activations_mlp = baseline_activations_mlp.sum(dim=0)
            # baseline_activations_hidd = self.get_baseline_with_activations(encoded_input, layer_idx, mask_idx)
            # activations_hidd = baseline_activations_hidd.sum(dim=0)
            # add activation of mlp layer
            return activations_mlp.mean(dim=0)

        # if attribution_method == "grad":
        #     # start = query_tensors.size(1) - 1
        #     # end = query_tensors.size(1) + response_tensors.size(1) - 1
        #     # baseline_outputs, baseline_activations = self.get_baseline_with_activations_2(encoded_input, layer_idx, start, end)
        #     baseline_outputs = self.model(**encoded_input,output_hidden_states=True)
        #     # compute logits my self after the mlp output but fine for now keep only last hidden state
        #     logits =  baseline_outputs.logits
        #     mask_idx = -2
        #     logits = logits[:, mask_idx, :]
        #     probs = F.softmax(logits, dim=-1)
        #     prob = torch.gather(probs, dim=-1, index=response_tensors[:,mask_idx].unsqueeze(-1)).squeeze(-1)
        #     if self.args.hidden and self.args.mlp:
        #          scores_hidd = torch.autograd.grad(probs, baseline_outputs.hidden_states[-1],grad_outputs=torch.ones_like(probs), retain_graph=True)[0]
        #          last_layer_mlp = self.model.transformer.h[self.args.layer_index].mlp  # 11
        #          before_last_hidden_state = baseline_outputs.hidden_states[self.args.layer_index-1]  # 10
        #          mlp_output = last_layer_mlp.c_fc(before_last_hidden_state)
        #          scores_mlp = torch.autograd.grad(probs, mlp_output,grad_outputs=torch.ones_like(probs), retain_graph=True)[0] # mlp output
        #          return scores_hidd, scores_mlp
        #     if self.args.mlp  and self.args.hidden== False:
        #          last_layer_mlp = self.model.transformer.h[self.args.layer_index].mlp  # 11
        #          before_last_hidden_state = baseline_outputs.hidden_states[self.args.layer_index-1]  # 10
        #          mlp_output = last_layer_mlp.c_fc(before_last_hidden_state)
        #          scores_mlp = torch.autograd.grad(probs, mlp_output,grad_outputs=torch.ones_like(probs), retain_graph=True)[0] # mlp output
        #          return scores_mlp
        #     if self.args.hidden and self.args.mlp== False:
        #          scores_hidd = torch.autograd.grad(probs, baseline_outputs.hidden_states[-1],grad_outputs=torch.ones_like(probs), retain_graph=True)[0] # mlp output
        #          return scores_hidd
        
        if attribution_method == "grad":
            # start = query_tensors.size(1) - 1
            # end = query_tensors.size(1) + response_tensors.size(1) - 1
            # baseline_outputs, baseline_activations = self.get_baseline_with_activations_2(encoded_input, layer_idx, start, end)
            # baseline_outputs = self.model(**encoded_input,output_hidden_states=True)
            # compute logits my self after the mlp output but fine for now keep only last hidden state
            logits =  baseline_outputs.logits
            mask_idx = -2
            logits = logits[:, mask_idx, :]
            probs = F.softmax(logits, dim=-1)
            prob = torch.gather(probs, dim=-1, index=response_tensors[:,mask_idx].unsqueeze(-1)).squeeze(-1)
            if self.args.hidden and self.args.mlp:
                 scores_hidd = torch.autograd.grad(probs, baseline_outputs.hidden_states[layer_idx],grad_outputs=torch.ones_like(probs), retain_graph=True)[0]
                 last_layer_mlp = self.model.transformer.h[layer_idx].mlp  # 11
                 before_last_hidden_state = baseline_outputs.hidden_states[layer_idx-1]  # 10
                 mlp_output = last_layer_mlp.c_fc(before_last_hidden_state)
                 scores_mlp = torch.autograd.grad(probs, mlp_output,grad_outputs=torch.ones_like(probs), retain_graph=True)[0] # mlp output
                 return scores_hidd, scores_mlp
            if self.args.mlp  and self.args.hidden== False:
                 last_layer_mlp = self.model.transformer.h[layer_idx].mlp  # 11
                 before_last_hidden_state = baseline_outputs.hidden_states[layer_idx-1]  # 10
                 mlp_output = last_layer_mlp.c_fc(before_last_hidden_state)
                 scores_mlp = torch.autograd.grad(probs, mlp_output,grad_outputs=torch.ones_like(probs), retain_graph=True)[0] # mlp output
                 return scores_mlp
            if self.args.hidden and self.args.mlp== False:
                 scores_hidd = torch.autograd.grad(probs, baseline_outputs.hidden_states[layer_idx],grad_outputs=torch.ones_like(probs), retain_graph=True)[0] # mlp output
                 return scores_hidd

        if attribution_method == "nothing":
            if self.args.hidden and self.args.mlp:
                scores_hidd = torch.rand(self.args.student_hidd_size)
                scores_mlp = torch.rand(self.args.student_mlp_size)
                return scores_hidd, scores_mlp
            elif self.args.hidden== True and self.args.mlp==False:
                return torch.rand(self.args.student_hidd_size)
            elif self.args.hidden== False and self.args.mlp==True:
                return torch.rand(self.args.student_mlp_size)


        if attribution_method == "gradX":
            # Forward pass
            # baseline_outputs = self.model(**encoded_input, output_hidden_states=True)
            logits = baseline_outputs.logits
            mask_idx = -2  # Index of the target token in the sequence
            logits = logits[:, mask_idx, :]
            probs = F.softmax(logits, dim=-1)
            prob = torch.gather(probs, dim=-1, index=response_tensors[:, mask_idx].unsqueeze(-1)).squeeze(-1)

            # Attribution method: Gradient × Input
            if self.args.hidden and self.args.mlp:
                # Hidden state attribution
                hidden_states = baseline_outputs.hidden_states[layer_idx]
                scores_hidd = torch.autograd.grad(prob, hidden_states, grad_outputs=torch.ones_like(prob), retain_graph=True)[0]
                grad_times_hidden = scores_hidd * hidden_states  # Gradient × Input for hidden states

                # MLP output attribution
                last_layer_mlp = self.model.transformer.h[layer_idx].mlp
                before_last_hidden_state = baseline_outputs.hidden_states[layer_idx - 1]
                mlp_output = last_layer_mlp.c_fc(before_last_hidden_state)
                scores_mlp = torch.autograd.grad(prob, mlp_output, grad_outputs=torch.ones_like(prob), retain_graph=True)[0]
                grad_times_mlp = scores_mlp * mlp_output  # Gradient × Input for MLP output

                return grad_times_hidden.abs(), grad_times_mlp.abs()

            if self.args.mlp and not self.args.hidden:
                # MLP output attribution only
                last_layer_mlp = self.model.transformer.h[layer_idx].mlp
                before_last_hidden_state = baseline_outputs.hidden_states[layer_idx - 1]
                mlp_output = last_layer_mlp.c_fc(before_last_hidden_state)
                scores_mlp = torch.autograd.grad(prob, mlp_output, grad_outputs=torch.ones_like(prob), retain_graph=True)[0]
                grad_times_mlp = scores_mlp * mlp_output  # Gradient × Input for MLP output
                return grad_times_mlp.abs()

            if self.args.hidden and not self.args.mlp:
                # Hidden state attribution only
                hidden_states = baseline_outputs.hidden_states[layer_idx]
                scores_hidd = torch.autograd.grad(prob, hidden_states, grad_outputs=torch.ones_like(prob), retain_graph=True)[0]
                grad_times_hidden = scores_hidd * hidden_states  # Gradient × Input for hidden states
                return grad_times_hidden.abs().sum(dim=0).sum(dim=0)

            
        if attribution_method == "gradXmask":
            # Forward pass
            # baseline_outputs = self.model(**encoded_input, output_hidden_states=True)
            logits = baseline_outputs.logits
            mask_idx = -2  # Index of the target token in the sequence
            logits = logits[:, mask_idx, :]
            probs = F.softmax(logits, dim=-1)
            prob = torch.gather(probs, dim=-1, index=response_tensors[:, mask_idx].unsqueeze(-1)).squeeze(-1)

            # Attribution method: Gradient × Input
            if self.args.hidden and self.args.mlp:
                # Hidden state attribution
                hidden_states = baseline_outputs.hidden_states[layer_idx]
                scores_hidd = torch.autograd.grad(prob, hidden_states, grad_outputs=torch.ones_like(prob), retain_graph=True)[0]
                grad_times_hidden = scores_hidd * hidden_states  # Gradient × Input for hidden states

                # MLP output attribution
                last_layer_mlp = self.model.transformer.h[layer_idx].mlp
                before_last_hidden_state = baseline_outputs.hidden_states[layer_idx - 1]
                mlp_output = last_layer_mlp.c_fc(before_last_hidden_state)
                scores_mlp = torch.autograd.grad(prob, mlp_output, grad_outputs=torch.ones_like(prob), retain_graph=True)[0]
                grad_times_mlp = scores_mlp * mlp_output  # Gradient × Input for MLP output

                return grad_times_hidden.abs(), grad_times_mlp.abs()

            if self.args.mlp and not self.args.hidden:
                # MLP output attribution only
                last_layer_mlp = self.model.transformer.h[layer_idx].mlp
                before_last_hidden_state = baseline_outputs.hidden_states[layer_idx - 1]
                mlp_output = last_layer_mlp.c_fc(before_last_hidden_state)
                scores_mlp = torch.autograd.grad(prob, mlp_output, grad_outputs=torch.ones_like(prob), retain_graph=True)[0]
                grad_times_mlp = scores_mlp * mlp_output  # Gradient × Input for MLP output
                
                return grad_times_mlp.abs()

            if self.args.hidden and not self.args.mlp:
                # Hidden state attribution only
                
                hidden_states = baseline_outputs.hidden_states[layer_idx]#[:,mask_idx,:]
                scores_hidd = torch.autograd.grad(prob, hidden_states, grad_outputs=torch.ones_like(prob), retain_graph=True)[0]
                grad_times_hidden = scores_hidd[:,mask_idx,:] * hidden_states[:,mask_idx,:]  # Gradient × Input for hidden states
                return grad_times_hidden.abs().sum(dim=0)

            
        if attribution_method == "gradlogitlens":
            # Forward pass
            # baseline_outputs = self.model(**encoded_input, output_hidden_states=True)
            # logits = baseline_outputs.logits
            # # mask_idx = -2  # Index of the target token in the sequence
            # # logits = logits[:, mask_idx, :]
            # probs = F.softmax(logits, dim=-1)
            # prob = torch.gather(probs, dim=-1, index=response_tensors[:, mask_idx].unsqueeze(-1)).squeeze(-1)

            # Attribution method: Gradient × Input
            if self.args.hidden and self.args.mlp:
                # Hidden state attribution
                baseline_outputs = self.model(**encoded_input, output_hidden_states=True)
                logits = baseline_outputs.logits
                mask_idx = -2  # Index of the target token in the sequence
                logits = logits[:, mask_idx, :]
                probs = F.softmax(logits, dim=-1)
                prob = torch.gather(probs, dim=-1, index=response_tensors[:, mask_idx].unsqueeze(-1)).squeeze(-1)
                hidden_states = baseline_outputs.hidden_states[layer_idx]
                scores_hidd = torch.autograd.grad(prob, hidden_states, grad_outputs=torch.ones_like(prob), retain_graph=True)[0]
                grad_times_hidden = scores_hidd * hidden_states  # Gradient × Input for hidden states

                # MLP output attribution
                last_layer_mlp = self.model.transformer.h[layer_idx].mlp
                before_last_hidden_state = baseline_outputs.hidden_states[layer_idx - 1]
                mlp_output = last_layer_mlp.c_fc(before_last_hidden_state)
                scores_mlp = torch.autograd.grad(prob, mlp_output, grad_outputs=torch.ones_like(prob), retain_graph=True)[0]
                grad_times_mlp = scores_mlp * mlp_output  # Gradient × Input for MLP output

                return grad_times_hidden.abs(), grad_times_mlp.abs()

            if self.args.mlp and not self.args.hidden:
                # MLP output attribution only
                baseline_outputs = self.model(**encoded_input, output_hidden_states=True)
                logits = baseline_outputs.logits
                mask_idx = -2  # Index of the target token in the sequence
                logits = logits[:, mask_idx, :]
                probs = F.softmax(logits, dim=-1)
                prob = torch.gather(probs, dim=-1, index=response_tensors[:, mask_idx].unsqueeze(-1)).squeeze(-1)
                last_layer_mlp = self.model.transformer.h[layer_idx].mlp
                before_last_hidden_state = baseline_outputs.hidden_states[layer_idx - 1]
                mlp_output = last_layer_mlp.c_fc(before_last_hidden_state)
                scores_mlp = torch.autograd.grad(prob, mlp_output, grad_outputs=torch.ones_like(prob), retain_graph=True)[0]
                grad_times_mlp = scores_mlp * mlp_output  # Gradient × Input for MLP output
                
                return grad_times_mlp.abs()

            if self.args.hidden and not self.args.mlp:
                # Hidden state attribution only
                #baseline_outputs = self.model(**encoded_input, output_hidden_states=True)
                start = query_tensors.size(1) - 1
                end = query_tensors.size(1) + response_tensors.size(1) - 1
                # Get hidden states for the specified layer
                mask_idx = -2
                hidden_states = baseline_outputs.hidden_states[layer_idx]#[:, mask_idx,:]
                # Logits computation from hidden states (model's last linear layer)
                # Assuming the model's final output layer maps hidden states to logits.
                # In GPT-2, this is done using the lm_head layer (which is a linear layer).
                # lm_head = self.model.lm_head  # the linear layer to convert hidden states to logits
                # logits = lm_head(hidden_states)  # shape: (batch_size, sequence_length, vocab_size)
                logits = baseline_outputs.logits
                # Get the logits for the target token
                # mask_idx = -2  # Index of the target token in the sequence (you can adjust this index)
                # logits = logits[:, mask_idx, :]  # Get logits for the masked token
                # probs = F.softmax(logits[:,start:end,:], dim=-1)  # Convert logits to probabilities
                # Extract the probability for the target token (given in response_tensors)
                curr_grad = torch.zeros_like(hidden_states)
                # for mask_idx in [start, int((end-start)/2), -2, -1]:
                for mask_idx in [-2]:
                #for mask_idx in [0, 1, 50, 400,-3, -2, -1]:
                    #logits = logits[:, mask_idx, :]  # Get logits for the masked token
                    probs = F.softmax(logits[:, mask_idx, :], dim=-1)  # Convert logits to probabilities
                    max_indices = torch.argmax(probs, dim=-1)
                    #prob = probs[:, max_indices]
                    #prob = torch.gather(probs, dim=-1, index=response_tensors[:, mask_idx].unsqueeze(-1)).squeeze(-1)
                    prob = torch.gather(probs, dim=-1, index=max_indices.unsqueeze(-1)).squeeze(-1)
                    #prob = torch.gather(probs, dim=-1, index=encoded_input["input_ids"][:, mask_idx].unsqueeze(-1)).squeeze(-1)
                    # Compute the gradients of the probability with respect to the hidden states
                    #hidden_states.requires_grad_()  # Set hidden states to require gradients
                    # Perform backward pass to compute gradients
                    # grads = torch.autograd.grad(prob, hidden_states, grad_outputs=torch.ones_like(prob), retain_graph=True)[0]
                    grads = torch.autograd.grad(torch.unbind(prob), hidden_states, retain_graph=True)[0]
                   
                    grad_times_hidden = grads * hidden_states
                    curr_grad = curr_grad + grad_times_hidden
                curr_grad = curr_grad/3
                # Gradient × Input (Gradients multiplied by the hidden states)
                # grad_times_hidden = grads * hidden_states  # Element-wise multiplication``
                return curr_grad.abs()  #.sum(dim=0).sum(dim=0)
                # hidden_states = baseline_outputs.hidden_states[layer_idx]#[:,mask_idx,:]
                # scores_hidd = torch.autograd.grad(prob, hidden_states, grad_outputs=torch.ones_like(prob), retain_graph=True)[0]
                # grad_times_hidden = scores_hidd[:,mask_idx,:] * hidden_states[:,mask_idx,:]  # Gradient × Input for hidden states
                # return grad_times_hidden.abs().sum(dim=0)
