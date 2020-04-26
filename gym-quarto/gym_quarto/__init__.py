# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 07:16:56 2020

@author: HP
"""

from gym.envs.registration import register

register(
    id='quarto-v0',
    entry_point='gym_quarto.envs:QuartoEnv',
)