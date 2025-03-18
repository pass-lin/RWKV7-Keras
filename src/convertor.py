# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 18:20:48 2025

@author: 路过的小林
"""
def convert_cmix(my_chnnal_mix,standard_chnnal_mix):
    key_weights = standard_chnnal_mix.key.weight.detach().cpu().T
    value_weights = standard_chnnal_mix.value.weight.detach().cpu().T
    xk_weights = standard_chnnal_mix.x_k.detach().cpu()
    my_chnnal_mix.set_weights([xk_weights,key_weights,value_weights,])
    
def convert_tmix(my_time_mix,standard_time_mix):
    #权重同步
    weights= [
        standard_time_mix.x_r.detach().cpu(),
        standard_time_mix.x_w.detach().cpu(),
        standard_time_mix.x_k.detach().cpu(),
        standard_time_mix.x_v.detach().cpu(),
        standard_time_mix.x_a.detach().cpu(),
        standard_time_mix.x_g.detach().cpu(),
        
        standard_time_mix.w0.detach().cpu(),
        standard_time_mix.w1.detach().cpu(),
        standard_time_mix.w2.detach().cpu(),
        
        standard_time_mix.a0.detach().cpu(),
        standard_time_mix.a1.detach().cpu(),
        standard_time_mix.a2.detach().cpu(),
        
        standard_time_mix.v0.detach().cpu(),
        standard_time_mix.v1.detach().cpu(),
        standard_time_mix.v2.detach().cpu(),
        
        standard_time_mix.g1.detach().cpu(),
        standard_time_mix.g2.detach().cpu(),
        
        standard_time_mix.k_k.detach().cpu(),
        standard_time_mix.k_a.detach().cpu(),
        standard_time_mix.r_k.detach().cpu(),
        
        standard_time_mix.receptance.weight.detach().cpu().T,
        standard_time_mix.key.weight.detach().cpu().T,
        standard_time_mix.value.weight.detach().cpu().T,
        standard_time_mix.output.weight.detach().cpu().T,
        
        standard_time_mix.ln_x.weight.detach().cpu(),
        standard_time_mix.ln_x.bias.detach().cpu(),
        
        ]
    my_time_mix.set_weights(weights)