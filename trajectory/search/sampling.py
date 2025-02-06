import numpy as np
import torch
import pdb

import matplotlib.pyplot as plt
# from julia import Main as JL
import pyjuice as juice
import torch.nn.functional as F
import time
from copy import deepcopy

#-------------------------------- helper functions --------------------------------#


action_filter_flag = False
state_filter_flag = True


def top_k_logits(logits, k, is_logits=True):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    if is_logits:
        out[out < v[:, [-1]]] = -float('Inf')
    else:
        out[out < v[:, [-1]]] = 0
        out = out / torch.sum(out,dim=1,keepdim=True)
    return out

def filter_cdf(logits, threshold, is_logits=True):
    batch_inds = torch.arange(logits.shape[0], device=logits.device, dtype=torch.long)
    bins_inds = torch.arange(logits.shape[-1], device=logits.device)
    if is_logits:
        probs = logits.softmax(dim=-1)
    else:
        probs = logits
    probs_sorted, _ = torch.sort(probs, dim=-1)
    probs_cum = torch.cumsum(probs_sorted, dim=-1)
    ## get minimum probability p such that the cdf up to p is at least `threshold`
    mask = probs_cum < threshold
    masked_inds = torch.argmax(mask * bins_inds, dim=-1)
    # pdb.set_trace()
    probs_threshold = probs_sorted[batch_inds, masked_inds]
    ## filter
    out = logits.clone()
    logits_mask = probs <= probs_threshold.unsqueeze(dim=-1)
    if is_logits:
        out[logits_mask] = -1000
    else:
        out[logits_mask] = 0
        out = out / torch.sum(out,dim=1,keepdim=True)
    return out

def round_to_multiple(x, N):
    '''
        Rounds `x` up to nearest multiple of `N`.

        x : int
        N : int
    '''
    pad = (N - x % N) % N
    return x + pad

def sort_2d(x):
    '''
        x : [ M x N ]
    '''
    M, N = x.shape
    x = x.view(-1)
    x_sort, inds = torch.sort(x, descending=True)

    rows = inds // N
    cols = inds % N

    return x_sort, rows, cols


def get_cropped_x(model, x, max_block=None, allow_crop=True, crop_increment=None, **kwargs):
    block_size = min(model.get_block_size(), max_block or np.inf)
    if x.shape[1] > block_size:
        n_crop = round_to_multiple(x.shape[1] - block_size, crop_increment)
        x = x[:, n_crop:]
    return x


#-------------------------------- forward pass --------------------------------#

def forward(model, x, max_block=None, allow_crop=True, crop_increment=None, **kwargs):
    '''
        A wrapper around a single forward pass of the transformer.
        Crops the input if the sequence is too long.

        x : tensor[ batch_size x sequence_length ]
    '''
    model.eval()
    block_size = min(model.get_block_size(), max_block or np.inf)
    # if x.shape[1] % 16 == 15:
    #     print(x[0,15::16])

    if x.shape[1] > block_size:
        assert allow_crop, (
            f'[ search/sampling ] input size is {x.shape} and block size is {block_size}, '
            'but cropping not allowed')

        ## crop out entire transition at a time so that the first token is always s_t^0

        n_crop = round_to_multiple(x.shape[1] - block_size, crop_increment)
        assert n_crop % crop_increment == 0
        x = x[:, n_crop:]
        
    # if x.shape[1] % 16 == 15:
    #     print(x[0,15::16])
    #     pdb.set_trace()
    logits, _ = model(x, **kwargs)

    return logits

def get_logp(model, x, temperature=1.0, topk=None, cdf=None, **forward_kwargs):
    '''
        x : tensor[ batch_size x sequence_length ]
    '''
    ## [ batch_size x sequence_length x vocab_size ]
    x_sft = torch.cat((torch.zeros([x.size(0), 1], dtype = torch.long, device = x.device), x), dim = 1)
    logits = forward(model, x_sft, **forward_kwargs)

    ## pluck the logits at the final step and scale by temperature
    ## [ batch_size x vocab_size ]
    logits = logits[:, -1] / temperature

    ## optionally crop logits to only the top `1 - cdf` percentile
    if cdf is not None:
        logits = filter_cdf(logits, cdf)

    ## optionally crop logits to only the most likely `k` options
    if topk is not None:
        logits = top_k_logits(logits, topk)


    ## apply softmax to convert to probabilities
    logp = logits.log_softmax(dim=-1)

    return logp

#-------------------------------- sampling --------------------------------#
def sample_with_pc_ratio(t, plan_kwargs, n_, N, pc_buffer, model, x, temperature=1.0, topk=None, cdf=None, **forward_kwargs):
    pc, [cache, all_fw_scopes, bk_scopes, val_probs] = pc_buffer
    
    num_vars = model.block_size + 1
    if num_vars % 16 == 0:
        action_dim = 3
    elif num_vars % 25 == 0:
        action_dim = 6
    crop_increment = forward_kwargs['crop_increment']
    top_ratio = plan_kwargs['top_ratio']
    pc_len = plan_kwargs['pc_len']
    dataset = plan_kwargs['dataset']

    if pc_len == 1 and (dataset != "halfcheetah-medium-replay-v2"):
        num_cats = model.vocab_size
    else:
        num_cats = model.vocab_size + 1


    x = get_cropped_x(model, x, **forward_kwargs)
    logits = forward(model, x, **forward_kwargs)
    logits = logits[:, -1] / temperature
    raw_probs = logits.softmax(dim=-1)



    val_idx = torch.arange(crop_increment-1,crop_increment*pc_len,crop_increment)
    rwd_idx = val_idx - 1 
    if N == action_dim and t == 0:
        delta = 1 - top_ratio
        if pc_len == 1:
            n_crop = round_to_multiple(x.shape[1] - crop_increment, crop_increment)
        else:
            # assert x.shape[1] > crop_increment
            n_crop = round_to_multiple(x.shape[1] - 2*crop_increment, crop_increment)
        x = x[:, n_crop:]
        num_missing_vars = crop_increment*pc_len - x.shape[1]
        pad = torch.zeros((x.shape[0],num_missing_vars),dtype=torch.int64).to(x.device)
        x_pad = torch.cat([x,pad],dim=1)
        mask = torch.ones((x_pad.shape[0]*num_cats,crop_increment*pc_len),dtype=torch.bool).to(x.device)
        mask[:,:x.shape[1]] = False
        mask[:,rwd_idx] = True
        mask[:,val_idx] = True
        eval_idx = torch.arange(num_cats)
        num_cats_ = len(eval_idx)

    

        if n_ == 0:
            pc.disable_partial_evaluation(forward = True, backward = True)
            if x.shape[1] < crop_increment:
                val_probs, cache = juice.queries.conditional(pc, target_vars=[crop_increment-1], tokens=x_pad.repeat(2,1), missing_mask=mask[:(x_pad.shape[0]*2)],cache={})
            else:
                val_probs, cache = juice.queries.conditional(pc, target_vars=[2*crop_increment-1], tokens=x_pad.repeat(2,1), missing_mask=mask[:(x_pad.shape[0]*2)],cache={})
            for key in cache.keys():
                cache[key] = cache[key][:,:x.shape[0]].repeat_interleave(num_cats_,dim=1)
            val_probs = val_probs[:x.shape[0],0,:]
        

        cum_val_probs = torch.cumsum(val_probs, dim=1)
        val_filter = (cum_val_probs < delta).repeat_interleave(num_cats_,dim=0)
        x_pad = x_pad.repeat_interleave(num_cats_,dim=0)
        x_pad[:,x.shape[1]] = eval_idx.repeat(x.shape[0])
        mask[:,x.shape[1]] = False

  
        if x.shape[1] < crop_increment:
            v_probs, cache = juice.queries.conditional(pc, target_vars=[crop_increment-1], tokens=x_pad, missing_mask=mask,
                                                        cache=cache, fw_delta_vars=[x.shape[1]], fw_scopes=all_fw_scopes[n_], bk_scopes=bk_scopes[0])
        else:
            v_probs, cache = juice.queries.conditional(pc, target_vars=[2*crop_increment-1], tokens=x_pad, missing_mask=mask,
                                                        cache=cache, fw_delta_vars=[x.shape[1]], fw_scopes=all_fw_scopes[n_+action_dim], bk_scopes=bk_scopes[1])
    

        v_probs = v_probs[:,0,:]
        val_probs = v_probs.clone()
        v_probs[val_filter] = 0.0 
        pc_ratio = v_probs.sum(1).reshape(x.shape[0],-1)
        raw_probs = raw_probs[:,:pc_ratio.shape[1]]
        raw_probs = raw_probs * pc_ratio
        raw_probs = raw_probs / torch.sum(raw_probs,dim=1,keepdim=True)



    probs = raw_probs.clone()
    if cdf is not None:
        probs = filter_cdf(probs, cdf, is_logits=False)

    if topk is not None:
        probs = top_k_logits(probs, topk, is_logits=False)
    

    indices = torch.multinomial(probs, num_samples=1)

    if N == action_dim and t == 0:
        for key in cache.keys():
            cache[key] = cache[key].reshape(-1,x.shape[0],num_cats_)[:,torch.arange(x.shape[0]).to(x.device),indices.reshape(-1)].repeat_interleave(num_cats_,dim=1)
        pc_buffer[1][0] = cache
        val_probs = val_probs.reshape(x.shape[0],num_cats_,num_cats_)[torch.arange(x.shape[0]).to(x.device),indices.reshape(-1),:]
        pc_buffer[1][3] = val_probs
    

    return indices, raw_probs, pc_buffer





def sample(N, n, ppc, model, x, temperature=1.0, topk=None, cdf=None, **forward_kwargs):
    '''
        Samples from the distribution parameterized by `model(x)`.

        x : tensor[ batch_size x sequence_length ]

        
    '''
    logits = forward(model, x, **forward_kwargs)
    logits = logits[:, -1] / temperature
    raw_probs = logits.softmax(dim=-1)


    if cdf is not None:
        logits = filter_cdf(logits, cdf)
    if topk is not None:
        logits = top_k_logits(logits, topk)
    probs = logits.softmax(dim=-1)

    indices = torch.multinomial(probs, num_samples=1)

    return indices, raw_probs, probs

@torch.no_grad()
def sample_n(t, plan_kwargs, value_fn, pc_buffer, model, x, N, **sample_kwargs):
    batch_size = len(x)

    probs = torch.zeros(batch_size, N, model.vocab_size + 1, device=x.device)

    for n in range(N): 
        indices, p, pc_buffer = sample_with_pc_ratio(t, plan_kwargs, n, N, pc_buffer, model, x, **sample_kwargs)
        x = torch.cat((x, indices), dim=1)
        probs[:, n, :p.shape[1]] = p

    return x, probs



