import json
import pdb
from os.path import join
import os
import torch
import numpy as np
import time

import trajectory.utils as utils
import trajectory.datasets as datasets
from trajectory.search import (
    beam_plan,
    make_prefix,
    extract_actions,
    update_context,
)

import pyjuice as juice
from pyjuice.nodes.methods import get_subsumed_scopes
from huggingface_hub import hf_hub_download

class Parser(utils.Parser):
    dataset: str = 'halfcheetah-medium-expert-v2'
    config: str = 'config.plan_conf'

#######################
######## setup ########
#######################

args = Parser().parse_args('plan')


#######################
####### Load GPT ######
#######################

dataset = utils.load_from_config(args.logbase, args.dataset, args.gpt_loadpath,
        'data_config.pkl')

gpt, gpt_epoch = utils.load_model(args.logbase, args.dataset, args.gpt_loadpath,
        epoch=args.gpt_epoch, device=args.device)


#######################
####### Load PC #######
#######################
pc_path = join(args.logbase, args.dataset, args.pc_loadpath)
pc_file = f"pc-{args.dataset}.jpc"

if not os.path.exists(join(pc_path, pc_file)):
    _ = hf_hub_download(
        repo_id="liebenxj/pretrained_pc", 
        filename=pc_file,
        repo_type="model",
        local_dir=pc_path,
        local_dir_use_symlinks=False,
    )

ns = juice.io.load(join(pc_path, pc_file))
print("> Compiling PC...")
pc_buffer = juice.TensorCircuit(ns)
device = torch.device("cuda:0")
pc_buffer.to(device)




#######################
####### dataset #######
#######################

env = datasets.load_environment(args.dataset)
timer = utils.timer.Timer()


#######################
####### set seed ######
#######################
seed_value = (int(args.suffix)+1)*10
env.seed(seed_value)
torch.manual_seed(seed_value)
np.random.seed(seed_value)

discretizer = dataset.discretizer
discount = dataset.discount
observation_dim = dataset.observation_dim
action_dim = dataset.action_dim

assert args.pc_len == 1
all_fw_scopes = []
all_bk_scopes = []
print("> Computing PC Scopes...")
crop_increment = observation_dim + action_dim + 2
val_idx = torch.arange(crop_increment-1,crop_increment*args.pc_len,crop_increment)
for j in val_idx:
    bk_scopes = get_subsumed_scopes(pc_buffer, [j.item()], type = "any")
    all_bk_scopes.append(bk_scopes)

for i in range(observation_dim,observation_dim+action_dim):
    fw_scopes = get_subsumed_scopes(pc_buffer, [i], type = "any")
    all_fw_scopes.append(fw_scopes)


for i in range(crop_increment+observation_dim,crop_increment+observation_dim+action_dim):
    fw_scopes = get_subsumed_scopes(pc_buffer, [i], type = "any")
    all_fw_scopes.append(fw_scopes)

    
value_fn = lambda x: discretizer.value_fn(x, args.percentile)
preprocess_fn = datasets.get_preprocess_fn(env.name)

#######################
###### main loop ######
#######################

observation = env.reset()
total_reward = 0
reward = 0
context = []
T = env.max_episode_steps

for t in range(T):

    observation = preprocess_fn(observation)

    if t % args.plan_freq == 0:
        ## concatenate previous transitions and current observations to input to model
        prefix = make_prefix(discretizer, context, observation, args.prefix_context)

        ## sample sequence from model beginning with `prefix`
        cache = dict()
        val_probs = None
 
        sequence = beam_plan(
            [pc_buffer,[cache,all_fw_scopes,all_bk_scopes,val_probs]],
            gpt, value_fn, prefix,
            args.horizon, args.beam_width, args.n_expand, observation_dim, action_dim,
            discount, args.max_context_transitions, verbose=args.verbose,
            k_obs=args.k_obs, k_act=args.k_act, cdf_obs=args.cdf_obs, cdf_act=args.cdf_act,
            top_ratio=args.top_ratio, dataset=args.dataset, pc_len=args.pc_len
        )
 
    else:
        sequence = sequence[1:]

    ## [ horizon x transition_dim ] convert sampled tokens to continuous trajectory
    sequence_recon = discretizer.reconstruct(sequence)



    ## [ action_dim ] index into sampled trajectory to grab first action
    action = extract_actions(sequence_recon, observation_dim, action_dim, t=0)

    ## execute action in environment
    next_observation, reward, terminal, _ = env.step(action)



    ## update return
    total_reward += reward
    score = env.get_normalized_score(total_reward)

    ## update context 
    context = update_context(context, discretizer, observation, action, reward, args.max_context_transitions)



    print(
        f'[ plan ] t: {t} / {T} | r: {reward:.2f} | R: {total_reward:.2f} | score: {score:.4f} | '
        f'time: {timer():.2f} | {args.dataset} | {args.exp_name} | {args.suffix}\n'
    )

    if terminal: break

    observation = next_observation


json_path = join(args.savepath, 'rollout.json')
json_data = {'score': score, 'step': t, 'return': total_reward, 'term': terminal}
json.dump(json_data, open(json_path, 'w'), indent=2, sort_keys=True)


