import numpy as np
import torch
import pdb

from .. import utils
from .sampling import sample_n



REWARD_DIM = VALUE_DIM = 1

@torch.no_grad()
def beam_plan(
    pc_buffer,
    model, value_fn, x,
    n_steps, beam_width, n_expand,
    observation_dim, action_dim,
    discount=0.99, max_context_transitions=None,
    k_obs=None, k_act=None, k_rew=1,
    cdf_obs=None, cdf_act=None, cdf_rew=None,
    verbose=True, top_ratio=None, dataset=None,pc_len = None):

    '''
        x : tensor[ 1 x input_sequence_length ]
    '''
    transition_dim = observation_dim + action_dim + REWARD_DIM + VALUE_DIM
    max_block = max_context_transitions * transition_dim - 1 if max_context_transitions else None





    ## pass in max numer of tokens to sample function
    sample_kwargs = {
        'max_block': max_block,
        'crop_increment': transition_dim,
    }

    plan_kwargs = {
        "top_ratio": top_ratio,
        "dataset": dataset,
        "pc_len": pc_len,
    }

    ## repeat input for search
    x = x.repeat(beam_width, 1)

    ## construct reward and discount tensors for estimating values
    rewards = torch.zeros(beam_width, n_steps + 1, device=x.device)
    discounts = discount ** torch.arange(n_steps + 1, device=x.device)

    ## logging
    progress = utils.Progress(n_steps) if verbose else utils.Silent()





    for t in range(n_steps):
        ## repeat everything by `n_expand` before we sample actions
        x = x.repeat(n_expand, 1)
        rewards = rewards.repeat(n_expand, 1)

        ## sample actions
        x, _ = sample_n(t, plan_kwargs, value_fn,pc_buffer, model, x, action_dim, topk=k_act, cdf=cdf_act, **sample_kwargs)


        ## sample reward and value estimate
        x, r_probs = sample_n(t, plan_kwargs, value_fn, pc_buffer, model, x, REWARD_DIM + VALUE_DIM, topk=k_rew, cdf=cdf_rew, **sample_kwargs)




        ## optionally, use a percentile or mean of the reward and
        ## value distributions instead of sampled tokens
        r_t, V_t = value_fn(r_probs)


        ## update rewards tensor
        rewards[:, t] = r_t
        rewards[:, t+1] = V_t

        ## estimate values using rewards up to `t` and terminal value at `t`
        values = (rewards * discounts).sum(dim=-1)


        ## get `beam_width` best actions
        values, inds = torch.topk(values, beam_width)

        ## index into search candidates to retain `beam_width` highest-reward sequences
        x = x[inds]
        rewards = rewards[inds]

        ## sample next observation (unless we have reached the end of the planning horizon)
        if t < n_steps - 1:
            x, _ = sample_n(t, plan_kwargs, value_fn, pc_buffer, model, x, observation_dim, topk=k_obs, cdf=cdf_obs, **sample_kwargs)

        ## logging
        progress.update({
            'x': list(x.shape),
            'vmin': values.min(), 'vmax': values.max(),
            'rtmin': r_t.min(), 'rtmax': r_t.max(),
            'vtmin': V_t.min(), 'vtmax': V_t.max(),
            # 'discount': discount
        })

    progress.stamp()

    ## [ batch_size x (n_context + n_steps) x transition_dim ]
    x = x.view(beam_width, -1, transition_dim)

    ## crop out context transitions
    ## [ batch_size x n_steps x transition_dim ]
    x = x[:, -n_steps:]



    ## return best sequence
    argmax = values.argmax()
    best_sequence = x[argmax]



    return best_sequence

