# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 16:45:17 2021

@author: gautier
"""
from bayes_opt import BayesianOptimization
from pathlib import Path
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from resparam import resunet_param

def f_to_optimize(lr_ini,decay_factor,step_size):
    return resunet_param(300,lr_ini*1e-4,decay_factor,step_size)
pbounds = {
#    'epochs': (100,500),
    'lr_ini': (5,100),
    'decay_factor': (0.1,0.9),
    'step_size':(10,200),
    }



LOG_DIR = Path().absolute() / 'bayes_opt_logs'
LOG_DIR.mkdir(exist_ok=True)

optimizer = BayesianOptimization(
    f=f_to_optimize,
    pbounds=pbounds,
    random_state=1,
    )

filename = 'log_0.json'
logger = JSONLogger(path=str(LOG_DIR / filename))
optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

optimizer.probe(
    params={'lr_ini': 6,
            'decay_factor': 0.75,
            'step_size': 50},
    lazy=True,
    
)
print(pbounds)


# Will probe only the two points specified above
optimizer.maximize(init_points=1, n_iter=10)

#optimizer.maximize(init_points=3, n_iter=10)

# with open('training_resUnet-markers.log', "r") as f1:
#     last_line = f1.readlines()[-1]
# last_line=last_line.split(',')
# iou_value=last_line[2]
# iou_value=float(iou_value)
# print(type(iou_value))
# print(iou_value)
