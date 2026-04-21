import torch, tqdm, random, sys, os, numpy as np, matplotlib.pyplot as plt
from scipy.integrate import odeint
from numpy.random import uniform
from torch.utils.data import TensorDataset, DataLoader
from modules import DataGenerator


def _load_or_generate(gen, gen_set, ode_func_args, dist_func_args, cache_path):
    """Load an (input, output) tensor pair from an .npz cache or generate+save.

    Only used for datasets that are otherwise simulated on the fly. Keeps the
    full dataset in float32 torch tensors in the returned values.
    """
    if cache_path and os.path.exists(cache_path):
        data = np.load(cache_path)
        return torch.from_numpy(data['input']).float(), torch.from_numpy(data['output']).float()
    inp, out = gen.generate_trajectory(gen_set=gen_set, ode_func_args=ode_func_args, dist_func_args=dist_func_args)
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez(cache_path, input=inp.numpy(), output=out.numpy())
    return inp, out

def ode_charge5(y, t, q):
    # Extract variables
    x0, x1, x2, x3, x4, y0, y1, y2, y3, y4, u0, u1, u2, u3, u4, v0, v1, v2, v3, v4 = y
    dx0, dy0 = u0, v0
    dx1, dy1 = u1, v1
    dx2, dy2 = u2, v2
    dx3, dy3 = u3, v3
    dx4, dy4 = u4, v4
    # Object 0
    i, a, b, c, d = 0, 1, 2, 3, 4
    du0 = q[i]*q[a]*(y[i] - y[a])/(np.sqrt((y[i] - y[a])**2 + (y[i+5] - y[a+5])**2)+1E-9)**3 +\
          q[i]*q[b]*(y[i] - y[b])/(np.sqrt((y[i] - y[b])**2 + (y[i+5] - y[b+5])**2)+1E-9)**3 +\
          q[i]*q[c]*(y[i] - y[c])/(np.sqrt((y[i] - y[c])**2 + (y[i+5] - y[c+5])**2)+1E-9)**3 +\
          q[i]*q[d]*(y[i] - y[d])/(np.sqrt((y[i] - y[d])**2 + (y[i+5] - y[d+5])**2)+1E-9)**3
    dv0 = q[i]*q[a]*(y[i+5] - y[a+5])/(np.sqrt((y[i] - y[a])**2 + (y[i+5] - y[a+5])**2)+1E-9)**3 +\
          q[i]*q[b]*(y[i+5] - y[b+5])/(np.sqrt((y[i] - y[b])**2 + (y[i+5] - y[b+5])**2)+1E-9)**3 +\
          q[i]*q[c]*(y[i+5] - y[c+5])/(np.sqrt((y[i] - y[c])**2 + (y[i+5] - y[c+5])**2)+1E-9)**3 +\
          q[i]*q[d]*(y[i+5] - y[d+5])/(np.sqrt((y[i] - y[d])**2 + (y[i+5] - y[d+5])**2)+1E-9)**3
    # Object 1
    i, a, b, c, d = 1, 0, 2, 3, 4
    du1 = q[i]*q[a]*(y[i] - y[a])/(np.sqrt((y[i] - y[a])**2 + (y[i+5] - y[a+5])**2)+1E-9)**3 +\
          q[i]*q[b]*(y[i] - y[b])/(np.sqrt((y[i] - y[b])**2 + (y[i+5] - y[b+5])**2)+1E-9)**3 +\
          q[i]*q[c]*(y[i] - y[c])/(np.sqrt((y[i] - y[c])**2 + (y[i+5] - y[c+5])**2)+1E-9)**3 +\
          q[i]*q[d]*(y[i] - y[d])/(np.sqrt((y[i] - y[d])**2 + (y[i+5] - y[d+5])**2)+1E-9)**3
    dv1 = q[i]*q[a]*(y[i+5] - y[a+5])/(np.sqrt((y[i] - y[a])**2 + (y[i+5] - y[a+5])**2)+1E-9)**3 +\
          q[i]*q[b]*(y[i+5] - y[b+5])/(np.sqrt((y[i] - y[b])**2 + (y[i+5] - y[b+5])**2)+1E-9)**3 +\
          q[i]*q[c]*(y[i+5] - y[c+5])/(np.sqrt((y[i] - y[c])**2 + (y[i+5] - y[c+5])**2)+1E-9)**3 +\
          q[i]*q[d]*(y[i+5] - y[d+5])/(np.sqrt((y[i] - y[d])**2 + (y[i+5] - y[d+5])**2)+1E-9)**3
    # Object 2
    i, a, b, c, d = 2, 0, 1, 3, 4
    du2 = q[i]*q[a]*(y[i] - y[a])/(np.sqrt((y[i] - y[a])**2 + (y[i+5] - y[a+5])**2)+1E-9)**3 +\
          q[i]*q[b]*(y[i] - y[b])/(np.sqrt((y[i] - y[b])**2 + (y[i+5] - y[b+5])**2)+1E-9)**3 +\
          q[i]*q[c]*(y[i] - y[c])/(np.sqrt((y[i] - y[c])**2 + (y[i+5] - y[c+5])**2)+1E-9)**3 +\
          q[i]*q[d]*(y[i] - y[d])/(np.sqrt((y[i] - y[d])**2 + (y[i+5] - y[d+5])**2)+1E-9)**3
    dv2 = q[i]*q[a]*(y[i+5] - y[a+5])/(np.sqrt((y[i] - y[a])**2 + (y[i+5] - y[a+5])**2)+1E-9)**3 +\
          q[i]*q[b]*(y[i+5] - y[b+5])/(np.sqrt((y[i] - y[b])**2 + (y[i+5] - y[b+5])**2)+1E-9)**3 +\
          q[i]*q[c]*(y[i+5] - y[c+5])/(np.sqrt((y[i] - y[c])**2 + (y[i+5] - y[c+5])**2)+1E-9)**3 +\
          q[i]*q[d]*(y[i+5] - y[d+5])/(np.sqrt((y[i] - y[d])**2 + (y[i+5] - y[d+5])**2)+1E-9)**3
    # Object 3
    i, a, b, c, d = 3, 0, 1, 2, 4
    du3 = q[i]*q[a]*(y[i] - y[a])/(np.sqrt((y[i] - y[a])**2 + (y[i+5] - y[a+5])**2)+1E-9)**3 +\
          q[i]*q[b]*(y[i] - y[b])/(np.sqrt((y[i] - y[b])**2 + (y[i+5] - y[b+5])**2)+1E-9)**3 +\
          q[i]*q[c]*(y[i] - y[c])/(np.sqrt((y[i] - y[c])**2 + (y[i+5] - y[c+5])**2)+1E-9)**3 +\
          q[i]*q[d]*(y[i] - y[d])/(np.sqrt((y[i] - y[d])**2 + (y[i+5] - y[d+5])**2)+1E-9)**3
    dv3 = q[i]*q[a]*(y[i+5] - y[a+5])/(np.sqrt((y[i] - y[a])**2 + (y[i+5] - y[a+5])**2)+1E-9)**3 +\
          q[i]*q[b]*(y[i+5] - y[b+5])/(np.sqrt((y[i] - y[b])**2 + (y[i+5] - y[b+5])**2)+1E-9)**3 +\
          q[i]*q[c]*(y[i+5] - y[c+5])/(np.sqrt((y[i] - y[c])**2 + (y[i+5] - y[c+5])**2)+1E-9)**3 +\
          q[i]*q[d]*(y[i+5] - y[d+5])/(np.sqrt((y[i] - y[d])**2 + (y[i+5] - y[d+5])**2)+1E-9)**3
    # Object 4
    i, a, b, c, d = 4, 0, 1, 2, 3
    du4 = q[i]*q[a]*(y[i] - y[a])/(np.sqrt((y[i] - y[a])**2 + (y[i+5] - y[a+5])**2)+1E-9)**3 +\
          q[i]*q[b]*(y[i] - y[b])/(np.sqrt((y[i] - y[b])**2 + (y[i+5] - y[b+5])**2)+1E-9)**3 +\
          q[i]*q[c]*(y[i] - y[c])/(np.sqrt((y[i] - y[c])**2 + (y[i+5] - y[c+5])**2)+1E-9)**3 +\
          q[i]*q[d]*(y[i] - y[d])/(np.sqrt((y[i] - y[d])**2 + (y[i+5] - y[d+5])**2)+1E-9)**3
    dv4 = q[i]*q[a]*(y[i+5] - y[a+5])/(np.sqrt((y[i] - y[a])**2 + (y[i+5] - y[a+5])**2)+1E-9)**3 +\
          q[i]*q[b]*(y[i+5] - y[b+5])/(np.sqrt((y[i] - y[b])**2 + (y[i+5] - y[b+5])**2)+1E-9)**3 +\
          q[i]*q[c]*(y[i+5] - y[c+5])/(np.sqrt((y[i] - y[c])**2 + (y[i+5] - y[c+5])**2)+1E-9)**3 +\
          q[i]*q[d]*(y[i+5] - y[d+5])/(np.sqrt((y[i] - y[d])**2 + (y[i+5] - y[d+5])**2)+1E-9)**3
    # Return
    return dx0, dx1, dx2, dx3, dx4, dy0, dy1, dy2, dy3, dy4, du0, du1, du2, du3, du4, dv0, dv1, dv2, dv3, dv4

def ode_spring5(y, t, k, graph):
    # Extract variables
    x1, x2, x3, x4, x5, y1, y2, y3, y4, y5, u1, u2, u3, u4, u5, v1, v2, v3, v4, v5 = y
    dx1, dy1 = u1, v1
    dx2, dy2 = u2, v2
    dx3, dy3 = u3, v3
    dx4, dy4 = u4, v4
    dx5, dy5 = u5, v5
    # Object 1
    du1 = (graph[0, 1]*k*(x2-x1)+graph[0, 2]*k*(x3-x1)+graph[0, 3]*k*(x4-x1)+graph[0, 4]*k*(x5-x1))/np.sum(graph[0, :])
    dv1 = (graph[0, 1]*k*(y2-y1)+graph[0, 2]*k*(y3-y1)+graph[0, 3]*k*(y4-y1)+graph[0, 4]*k*(y5-y1))/np.sum(graph[0, :])
    # Object 2
    du2 = (graph[1, 0]*k*(x1-x2)+graph[1, 2]*k*(x3-x2)+graph[1, 3]*k*(x4-x2)+graph[1, 4]*k*(x5-x2))/np.sum(graph[1, :])
    dv2 = (graph[1, 0]*k*(y1-y2)+graph[1, 2]*k*(y3-y2)+graph[1, 3]*k*(y4-y2)+graph[1, 4]*k*(y5-y2))/np.sum(graph[1, :])
    # Object 3
    du3 = (graph[2, 0]*k*(x1-x3)+graph[2, 1]*k*(x2-x3)+graph[2, 3]*k*(x4-x3)+graph[2, 4]*k*(x5-x3))/np.sum(graph[2, :])
    dv3 = (graph[2, 0]*k*(y1-y3)+graph[2, 1]*k*(y2-y3)+graph[2, 3]*k*(y4-y3)+graph[2, 4]*k*(y5-y3))/np.sum(graph[2, :])
    # Object 4
    du4 = (graph[3, 0]*k*(x1-x4)+graph[3, 1]*k*(x2-x4)+graph[3, 2]*k*(x3-x4)+graph[3, 4]*k*(x5-x4))/np.sum(graph[3, :])
    dv4 = (graph[3, 0]*k*(y1-y4)+graph[3, 1]*k*(y2-y4)+graph[3, 2]*k*(y3-y4)+graph[3, 4]*k*(y5-y4))/np.sum(graph[3, :])
    # Object 5
    du5 = (graph[4, 0]*k*(x1-x5)+graph[4, 1]*k*(x2-x5)+graph[4, 2]*k*(x3-x5)+graph[4, 3]*k*(x4-x5))/np.sum(graph[4, :])
    dv5 = (graph[4, 0]*k*(y1-y5)+graph[4, 1]*k*(y2-y5)+graph[4, 2]*k*(y3-y5)+graph[4, 3]*k*(y4-y5))/np.sum(graph[4, :])
    # Return
    return dx1, dx2, dx3, dx4, dx5, dy1, dy2, dy3, dy4, dy5, du1, du2, du3, du4, du5, dv1, dv2, dv3, dv4, dv5

def ode_kuramoto5(y, t, k, graph, omega):
    # Extract variables
    x1, x2, x3, x4, x5 = y
    # Object 1
    dx1 = omega[0] + k*graph[0, 1]*np.sin(x1-x2) + k*graph[0, 2]*np.sin(x1-x3) + k*graph[0, 3]*np.sin(x1-x4) + k*graph[0, 4]*np.sin(x1-x5)
    # Object 2
    dx2 = omega[1] + k*graph[1, 0]*np.sin(x2-x1) + k*graph[1, 2]*np.sin(x2-x3) + k*graph[1, 3]*np.sin(x2-x4) + k*graph[1, 4]*np.sin(x2-x5)
    # Object 3
    dx3 = omega[2] + k*graph[2, 0]*np.sin(x3-x1) + k*graph[2, 1]*np.sin(x3-x2) + k*graph[2, 3]*np.sin(x3-x4) + k*graph[2, 4]*np.sin(x3-x5)
    # Object 4
    dx4 = omega[3] + k*graph[3, 0]*np.sin(x4-x1) + k*graph[3, 1]*np.sin(x4-x2) + k*graph[3, 2]*np.sin(x4-x3) + k*graph[3, 4]*np.sin(x4-x5)
    # Object 5
    dx5 = omega[4] + k*graph[4, 0]*np.sin(x5-x1) + k*graph[4, 1]*np.sin(x5-x2) + k*graph[4, 2]*np.sin(x5-x3) + k*graph[4, 3]*np.sin(x5-x4)
    # Return
    return dx1, dx2, dx3, dx4, dx5

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
    # print(output_train.shape)
    # print(output_train[0, 0::4].cpu().detach().numpy().shape)
    # plt.figure()
    # t = np.linspace(1, 49, 49)
    # print(t.shape)
    # plt.subplot(1, 2, 1)
    # plt.plot(t, output_train[0, 0::4].cpu().detach().numpy())
    # plt.plot(t, output_train[0, 1::4].cpu().detach().numpy())
    # plt.subplot(1, 2, 2)
    # plt.plot(t, output_train[0, 2::4].cpu().detach().numpy(), color='C0')
    # plt.plot(t, output_train[0, 3::4].cpu().detach().numpy(), color='C1')
    # plt.plot(np.linspace(1, 48, 48), -(output_train[0, 0:-4:4].cpu().detach().numpy() - output_train[0, 4::4].cpu().detach().numpy())/(dt*100), linestyle='dashed', color='C0', linewidth=3)
    # plt.plot(np.linspace(1, 48, 48), -(output_train[0, 1:-4:4].cpu().detach().numpy() - output_train[0, 5::4].cpu().detach().numpy())/(dt*100), linestyle='dashed', color='C1', linewidth=3)
    # plt.savefig('fig.png')
    # sys.exit()
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

def load_MS5(batch_size, tmax, dt, gen_set, plot_option, num_worker):
    # Initialise simulation parameters
    k = 2.5
    # graph = np.array([[1, 0, 1, 1, 0], [1, 0, 0, 0, 1], [0, 1, 1, 0, 1], [1, 0, 1, 0, 1], [1, 1, 0, 0, 1]])
    graph = np.array([[0, 1, 0, 1, 1], [1, 0, 0, 1, 1], [0, 0, 0, 0, 1], [1, 1, 0, 0, 1], [1, 1, 1, 1, 0]])
    ode_func_args = (k, graph)
    dist_func_args = [0.0, 0.3]
    # Initialise DataGenerator
    gen = DataGenerator(ode_func=ode_spring5, dist_func=np.random.normal, tmax=tmax, dt=dt, sampling_freq=100, onestep_dim=20, plot_option=plot_option, aug_option=False)
    # Cache generated trajectories under Data/MS5/ so repeat runs skip the
    # ~18-minute simulation step. Cache key includes gen_set for train/valid
    # because those sizes vary; test set size is fixed at 2500.
    cache_dir = os.path.join("Data", "MS5")
    train_cache = os.path.join(cache_dir, f"ms5_train_{gen_set}.npz")
    valid_cache = os.path.join(cache_dir, f"ms5_valid_{gen_set//4}.npz")
    test_cache  = os.path.join(cache_dir, "ms5_test_2500.npz")
    # Training dataset
    input_train, output_train = _load_or_generate(gen, gen_set, ode_func_args, dist_func_args, train_cache)
    train_dataset = TensorDataset(input_train, output_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_worker, pin_memory=True)
    # Validation dataset
    input_valid, output_valid = _load_or_generate(gen, int(gen_set/4), ode_func_args, dist_func_args, valid_cache)
    valid_dataset = TensorDataset(input_valid, output_valid)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_worker, pin_memory=True)
    # Testing dataset
    input_test, output_test = _load_or_generate(gen, 2500, ode_func_args, dist_func_args, test_cache)
    test_dataset = TensorDataset(input_test, output_test)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, drop_last=True, num_workers=num_worker, pin_memory=True)
    # Return
    return train_loader, valid_loader, test_loader

def load_KM5(batch_size, tmax, dt, gen_set, plot_option, num_worker):
    # Initialise simulation parameters
    k = 1.5
    # graph = np.array([[1, 0, 1, 1, 0], [1, 0, 0, 0, 1], [0, 1, 1, 0, 1], [1, 0, 1, 0, 1], [1, 1, 0, 0, 1]])
    graph = np.array([[0, 1, 0, 1, 1], [1, 0, 0, 1, 1], [0, 0, 0, 0, 1], [1, 1, 0, 0, 1], [1, 1, 1, 1, 0]])
    omega = np.array([2, 1, 5, 3, 9])
    ode_func_args = (k, graph, omega)
    dist_func_args = [0.0, 2*np.pi]
    # Initialise DataGenerator
    gen = DataGenerator(ode_func=ode_kuramoto5, dist_func=np.random.uniform, tmax=tmax, dt=dt, sampling_freq=10, onestep_dim=5, plot_option=plot_option, aug_option=True)
    # Generate training dataset
    state_t_train, state_tpn_train = gen.generate_trajectory(gen_set=gen_set, ode_func_args=ode_func_args, dist_func_args=dist_func_args)
    train_dataset = TensorDataset(state_t_train, state_tpn_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_worker, pin_memory=True)
    # Generate validation dataset
    state_t_valid, state_tpn_valid = gen.generate_trajectory(gen_set=int(gen_set/4), ode_func_args=ode_func_args, dist_func_args=dist_func_args)
    valid_dataset = TensorDataset(state_t_valid, state_tpn_valid)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_worker, pin_memory=True)
    # Generate testing dataset
    state_t_test, state_tpn_test = gen.generate_trajectory(gen_set=2500, ode_func_args=ode_func_args, dist_func_args=dist_func_args)
    test_dataset = TensorDataset(state_t_test, state_tpn_test)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, drop_last=True, num_workers=num_worker, pin_memory=True)
    # Return
    return train_loader, valid_loader, test_loader

def load_CH5(batch_size, num_worker):
    # Load charged particle data
    data = np.load("Data/Charged_Particles/Charged.npz")
    data = torch.tensor(data['state'], dtype=torch.float32)
    # Split data into input and output data
    input_data, output_data = data[:, :20], data[:, 20:]
    # Create dataset
    full_dataset = TensorDataset(input_data, output_data)
    # Create generator object for random split
    gen = torch.Generator()
    gen.manual_seed(72)
    # Splitting dataset into train, test, and validation sets
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [8/12, 2/12, 2/12], generator=gen)
    # Generate training dataset
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_worker, pin_memory=True)
    # Generate validation dataset
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_worker, pin_memory=True)
    # Generate testing dataset
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, drop_last=True, num_workers=num_worker, pin_memory=True)
    # Return
    return train_loader, valid_loader, test_loader

def load_TR5(batch_size, num_worker):
    # Read training dataset
    state_t_train, state_tpn_train = np.load("Data/TrajNet/TrajTrainInputAugmented.npz"), np.load("Data/TrajNet/TrajTrainOutputAugmented.npz")
    state_t_train, state_tpn_train = torch.tensor(state_t_train['arr_0'], dtype=torch.float32), torch.tensor(state_tpn_train['arr_0'], dtype=torch.float32)
    train_dataset = TensorDataset(state_t_train, state_tpn_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_worker, pin_memory=True)
    # Read validation dataset
    state_t_valid, state_tpn_valid = np.load("Data/TrajNet/TrajValidInputAugmented.npz"), np.load("Data/TrajNet/TrajValidOutputAugmented.npz")
    state_t_valid, state_tpn_valid = torch.tensor(state_t_valid['arr_0'], dtype=torch.float32), torch.tensor(state_tpn_valid['arr_0'], dtype=torch.float32)
    valid_dataset = TensorDataset(state_t_valid, state_tpn_valid)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_worker, pin_memory=True)
    # Read testing dataset
    state_t_test, state_tpn_test = np.load("Data/TrajNet/TrajTestInputAugmented.npz"), np.load("Data/TrajNet/TrajTestOutputAugmented.npz")
    state_t_test, state_tpn_test = torch.tensor(state_t_test['arr_0'], dtype=torch.float32), torch.tensor(state_tpn_test['arr_0'], dtype=torch.float32)
    test_dataset = TensorDataset(state_t_test, state_tpn_test)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, drop_last=True, num_workers=num_worker, pin_memory=True)
    # Return
    return train_loader, valid_loader, test_loader

def load_NBA(batch_size, num_worker):
    # Read training dataset
    # state_t_train, state_tpn_train = np.load("Data/NBA/datasets_truncated/train_input.npy"), np.load("Data/NBA/datasets_truncated/train_output.npy")
    state_t_train, state_tpn_train = np.load("datasets_old/train_input.npy"), np.load("datasets_old/train_output.npy")
    state_t_train, state_tpn_train = torch.tensor(state_t_train, dtype=torch.float32), torch.tensor(state_tpn_train, dtype=torch.float32)
    train_dataset = TensorDataset(state_t_train, state_tpn_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_worker, pin_memory=True)
    # Read validation dataset
    # state_t_valid, state_tpn_valid = np.load("Data/NBA/datasets_truncated/test_input.npy"), np.load("Data/NBA/datasets_truncated/test_output.npy")
    state_t_valid, state_tpn_valid = np.load("datasets_old/test_input.npy"), np.load("datasets_old/test_output.npy")
    state_t_valid, state_tpn_valid = torch.tensor(state_t_valid, dtype=torch.float32), torch.tensor(state_tpn_valid, dtype=torch.float32)
    valid_dataset = TensorDataset(state_t_valid, state_tpn_valid)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_worker, pin_memory=True)
    # Read testing dataset
    # state_t_test, state_tpn_test = np.load("Data/NBA/datasets_truncated/test_input.npy"), np.load("Data/NBA/datasets_truncated/test_output.npy")
    state_t_test, state_tpn_test = np.load("datasets_old/test_input.npy"), np.load("datasets_old/test_output.npy")
    state_t_test, state_tpn_test = torch.tensor(state_t_test, dtype=torch.float32), torch.tensor(state_tpn_test, dtype=torch.float32)
    test_dataset = TensorDataset(state_t_test, state_tpn_test)
    test_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=num_worker, pin_memory=True)
    # Return
    return train_loader, valid_loader, test_loader

def load_dataset(model_name, batch_size, train_set, plot_option, num_worker):
    if model_name == "DP2":
        nAgent, nDim, dt, data_offset = 2, 1, 0.05, 49
        train_loader, valid_loader, test_loader = load_DP2(batch_size=batch_size, tmax=5000, dt=dt/100, gen_set=train_set, plot_option=plot_option, num_worker=num_worker)
        return train_loader, valid_loader, test_loader, nAgent, nDim, dt, data_offset
    if model_name == "MS5":
        nAgent, nDim, dt, data_offset = 5, 2, 0.05, 49
        train_loader, valid_loader, test_loader = load_MS5(batch_size=batch_size, tmax=5000, dt=dt/100, gen_set=train_set, plot_option=plot_option, num_worker=num_worker)
        return train_loader, valid_loader, test_loader, nAgent, nDim, dt, data_offset
    if model_name == "KM5":
        nAgent, nDim, dt, data_offset = 5, 1, 0.05, 49
        train_loader, valid_loader, test_loader = load_KM5(batch_size=batch_size, tmax=500, dt=dt/10, gen_set=train_set, plot_option=plot_option, num_worker=num_worker)
        return train_loader, valid_loader, test_loader, nAgent, nDim, dt, data_offset
    if model_name == "CH5":
        nAgent, nDim, dt, data_offset = 5, 2, 0.1, 49
        train_loader, valid_loader, test_loader = load_CH5(batch_size=batch_size, num_worker=num_worker)
        return train_loader, valid_loader, test_loader, nAgent, nDim, dt, data_offset
    if model_name == "TR5":
        nAgent, nDim, dt, data_offset = 5, 2, 1.0, 19
        train_loader, valid_loader, test_loader = load_TR5(batch_size=batch_size, num_worker=num_worker)
        return train_loader, valid_loader, test_loader, nAgent, nDim, dt, data_offset
    if model_name == "NBA":
        nAgent, nDim, dt, data_offset = 11, 2, 0.1, 13
        train_loader, valid_loader, test_loader = load_NBA(batch_size=batch_size, num_worker=num_worker)
        return train_loader, valid_loader, test_loader, nAgent, nDim, dt, data_offset
