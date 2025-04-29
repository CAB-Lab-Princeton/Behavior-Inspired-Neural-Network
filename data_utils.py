import torch, tqdm, random, sys, numpy as np, matplotlib.pyplot as plt
from scipy.integrate import odeint
from numpy.random import uniform
from torch.utils.data import TensorDataset, DataLoader
from modules import DataGenerator

def ode_dp2(y, t, L1, L2, m1, m2):
    g = 9.81
    theta1, theta2, z1, z2 = y
    c, s = np.cos(theta1-theta2), np.sin(theta1-theta2)
    theta1dot = z1
    theta2dot = z2
    z1dot = (m2*g*np.sin(theta2)*c - m2*s*(L1*z1**2*c + L2*z2**2) -
             (m1+m2)*g*np.sin(theta1)) / L1 / (m1 + m2*s**2)
    z2dot = ((m1+m2)*(L1*z1**2*s - g*np.sin(theta2) + g*np.sin(theta1)*c) + 
             m2*L2*z2**2*s*c) / L2 / (m1 + m2*s**2)
    return theta1dot, theta2dot, z1dot, z2dot

def load_DP2(batch_size, tmax, dt, gen_set, plot_option, num_worker):
    # Initialise simulation parameters
    L1, L2, m1, m2  = 1, 1, 1, 1
    ode_func_args = (L1, L2, m1, m2)
    dist_func_args = [0.0, 0.5]
    # Initialise DataGenerator
    gen = DataGenerator(ode_func=ode_dp2, dist_func=np.random.normal, tmax=tmax, dt=dt, sampling_freq=100, onestep_dim=4, plot_option=plot_option, aug_option=False)
    # Generate training dataset
    input_train, output_train = gen.generate_trajectory(gen_set=gen_set, ode_func_args=ode_func_args, dist_func_args=dist_func_args)
    train_dataset = TensorDataset(input_train, output_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_worker, pin_memory=True)
    # Generate validation dataset
    input_valid, output_valid = gen.generate_trajectory(gen_set=int(gen_set/4), ode_func_args=ode_func_args, dist_func_args=dist_func_args)
    valid_dataset = TensorDataset(input_valid, output_valid)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_worker, pin_memory=True)
    # Generate testing dataset
    input_test, output_test = gen.generate_trajectory(gen_set=int(gen_set/4), ode_func_args=ode_func_args, dist_func_args=dist_func_args)
    test_dataset = TensorDataset(input_test, output_test)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, drop_last=True, num_workers=num_worker, pin_memory=True)
    # Return
    return train_loader, valid_loader, test_loader

def load_dataset(model_name, batch_size, train_set, plot_option, num_worker):
    if model_name == "DP2":
        nAgent, nDim, dt, data_offset = 2, 1, 0.05, 49
        train_loader, valid_loader, test_loader = load_DP2(batch_size=batch_size, tmax=5000, dt=dt/100, gen_set=train_set, plot_option=plot_option, num_worker=num_worker)
        return train_loader, valid_loader, test_loader, nAgent, nDim, dt, data_offset
