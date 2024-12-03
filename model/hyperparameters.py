import jax.numpy as jnp
from .model import ModelHyperparameters
from .training import TrainingHyperparameters
from .cortical_column import CorticalColumnHyperparameters


dt = 1e-4


cortical_column_hyperparameters = CorticalColumnHyperparameters(
    e0=5,
    r=0.7,
    s0=10,
    excitatory_gain=5.17,
    excitatory_time_constant=7.7,
    slow_inhibitory_gain=4.45,
    slow_inhibitory_time_constant=34,
    fast_inhibitory_gain=57.1,
    fast_inhibitory_time_constant=6.8,
    c_ep=31.7,
    c_pe=17.3,
    c_sp=51.9,
    c_ps=100,
    c_pp=300,
    c_fp=66.9,
    c_fs=100,
    c_pf=16,
    c_ff=18,
    var_p=5,
    var_f=5,
)

sequence_model_hyperparameters = ModelHyperparameters(
    w_l1_wm=100, w_wm_l1=100, w_l2_l1=120, w_l3_l2=186, t=20, r=1000
)

semantic_model_hyperparameters = ModelHyperparameters(
    w_l1_wm=300, w_wm_l1=300, w_l2_l1=120, w_l3_l2=186, t=20, r=1000
)

training_hyperparameters = TrainingHyperparameters(
    e0=5,
    theta_low1=0.12,
    theta_low2=0.8,
    theta_high2=0.6,
    theta_low3=0.7,
    gamma_w=0.1,
    gamma_k=1,
    gamma_a=1,
    gamma_wb=10,
    w_l1_l1_max=10,
    k_l2_l2_max=8,
    a_l2_l2_max=0.12,
    k_l3_l3_max=8,
    a_l3_l3_max=0.12,
    w_l2_l3_max=11,
    w_max_sum=130,
    k_max_sum=160,
    a_max_sum=None,
)

training_mask_1 = jnp.array([[0, 0], [1, 0], [1, 1], [1, 1]])
training_mask_2 = jnp.array([[0, 0], [0, 0], [1, 0], [1, 0]])

pattern_intensity = 2000
