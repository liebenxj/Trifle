from trajectory.utils import watch

## automatically make experiment names for planning
## by labelling folders with these args
args_to_watch = [
    ('prefix', ''),
    ('plan_freq', 'freq'),
    ('horizon', 'H'),
    ('beam_width', 'beam'),
]

base = {

    'plan': {
        'logbase': 'logs/',
        'gpt_loadpath': 'gpt/pretrained',
        'pc_loadpath': 'pc/pretrained',
        'gpt_epoch': 'latest',
        'device': 'cuda',
        'renderer': 'Renderer',

        'plan_freq': 1,
        'horizon': 5,
        'beam_width': 32,
        'n_expand': 2,

        'k_obs': 1,
        'k_act': None,
        'cdf_obs': None,
        'cdf_act': 0.6,
        'percentile': 'mean',

        'max_context_transitions': 5,
        'prefix_context': True,


        'exp_name': watch(args_to_watch),
        'prefix': 'plans/trifle/',
        'suffix': '0',
        'verbose': True,

        'top_ratio': 0.2, 
        "pc_len": 1,
    },
}



#------------------------ locomotion ------------------------#

## for all halfcheetah environments, you can reduce the planning horizon and beam width without
## affecting performance. good for speed and sanity.

halfcheetah_medium_v2 = halfcheetah_medium_replay_v2 = {
    'plan': {
        'horizon': 5,
        'beam_width': 32,
    }
}

halfcheetah_medium_expert_v2 = {
    'plan': {
        'beam_width': 32,
    },
}

## if you leave the dictionary empty, it will use the base parameters
hopper_medium_expert_v2 = hopper_medium_v2 = walker2d_medium_v2 = {}

## hopper and wlaker2d are a little more sensitive to planning hyperparameters; 
## proceed with caution when reducing the horizon or increasing the planning frequency

hopper_medium_replay_v2 = {
    'train': {
        ## train on the medium-replay datasets longer
        'n_epochs_ref': 80,
    },
}

walker2d_medium_expert_v2 = {
    'plan': {
        ## also safe to reduce the horizon here
        'horizon': 5,
    },
}

walker2d_medium_replay_v2 = {
    'train': {
        ## train on the medium-replay datasets longer
        'n_epochs_ref': 80,
    },
    'plan': {
        ## can reduce beam width, but need to adjust action sampling
        ## distribution and increase horizon to accomodate
        'horizon': 20,
        'beam_width': 32,
        'k_act': 40,
        'cdf_act': None,
    }
}

ant_medium_v2 = ant_medium_replay_v2 = ant_random_v2 = {
    'train': {
        ## reduce batch size because the dimensionality is larger
        'batch_size': 128,
    },
    'plan': {
        'horizon': 5,
    }
}
