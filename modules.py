import torch, tqdm, sys
from torch import nn
from scipy.integrate import odeint
import torch.nn.functional as F, numpy as np, matplotlib.pyplot as plt

def encode_onehot(labels):
    # From NRI paper, one-hot labeling for message passing
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    # Return
    return labels_onehot
    
class DLCMLP(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super(DLCMLP, self).__init__()
        # Initialise linear networks
        self.z_linear1 = nn.Linear(in_dim, hid_dim)
        self.z_linear2 = nn.Linear(hid_dim, hid_dim)
        self.z_linear3 = nn.Linear(hid_dim, out_dim)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
    
    def forward(self, x):
        z_pred = self.z_linear1(x)
        z_pred = F.tanh(z_pred)
        z_pred = self.z_linear2(z_pred)
        z_pred = F.tanh(z_pred)
        z_pred = self.z_linear3(z_pred)
        return z_pred

class NonlinearOpinionDynamics(nn.Module):
    def __init__(self, nAgent, nOpinion, nDim, dOrder, in_dim, hid_dim, device, arch="DLCMLP"):
        super(NonlinearOpinionDynamics, self).__init__()
        if arch == 'DLCMLP':
            self.b = DLCMLP(in_dim=in_dim, hid_dim=hid_dim, out_dim=nAgent*nOpinion)
        elif arch == 'MPNN':
            off_diag = np.ones([nAgent, nAgent]) - np.eye(nAgent)
            rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
            rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
            rel_rec = torch.FloatTensor(rel_rec).to(device)
            rel_send = torch.FloatTensor(rel_send).to(device)
            self.b = MPNN(nAgent=nAgent, in_dim=nDim*dOrder, hid_dim=hid_dim, out_dim=nOpinion, rel_rec=rel_rec, rel_send=rel_send)
        # Damping parameters
        self.d = torch.nn.Parameter(torch.distributions.Normal(0, 1).sample((nAgent, nOpinion)))
        # Self attention parameters
        self.alpha = torch.nn.Parameter(torch.distributions.Normal(0, 1).sample((nAgent, nOpinion)))
        # Social interaction parameters
        self.u = torch.nn.Parameter(torch.distributions.Normal(0, 1).sample((nAgent, 1)))
        # Belief graph parameters
        self.oGraph_upper = torch.nn.Parameter(torch.distributions.Normal(0, 1).sample((int(nOpinion*(nOpinion-1)/2),)))
        self.oGraph_upper_index = torch.triu_indices(nOpinion, nOpinion, 1)
        self.oGraph_lower = torch.nn.Parameter(torch.distributions.Normal(0, 1).sample((int(nOpinion*(nOpinion-1)/2),)))
        self.oGraph_lower_index = torch.tril_indices(nOpinion, nOpinion, -1)
        # Communication graph parameters
        self.aGraph_prefactor = torch.nn.Parameter(torch.distributions.Normal(0, 1).sample((nAgent, nAgent)))
    
class MPMLP(nn.Module):
    def __init__(self, n_in, n_hid, n_out, activation='tanh'):
        super(MPMLP, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hid)
        self.fc2 = nn.Linear(n_hid, n_hid)
        self.fc3 = nn.Linear(n_hid, n_out)
        if activation == 'tanh':
            self.activation = F.tanh
        elif activation == 'relu':
            self.activation = F.relu
        elif activation == 'leaky_relu':
            self.activation = F.leaky_relu
        elif activation == 'elu':
            self.activation = F.elu
        elif activation == 'gelu':
            self.activation = F.gelu
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.0)

    def forward(self, inputs):
        x = self.fc1(inputs)
        x = F.tanh(x)
        x = self.fc2(x)
        x = F.tanh(x)
        x = self.fc3(x)
        return x

class MPNN(nn.Module):
    def __init__(self, nAgent, in_dim, hid_dim, out_dim, rel_rec, rel_send, activation='tanh'):
        super(MPNN, self).__init__()
        self.nAgent = nAgent
        self.in_dim, self.hid_dim, self.out_dim = in_dim, hid_dim, out_dim
        self.MPMLP1 = MPMLP(in_dim, hid_dim, hid_dim, activation)
        self.MPMLP2 = MPMLP(2*hid_dim, hid_dim, hid_dim, activation)
        self.MPMLP3 = MPMLP(hid_dim, hid_dim, out_dim, activation)
        self.rel_rec = rel_rec
        self.rel_send = rel_send

    def edge2node(self, x):
        incoming = torch.matmul(self.rel_rec.t(), x)
        return incoming/incoming.size(1)

    def node2edge(self, x):
        receivers = torch.matmul(self.rel_rec, x)
        senders = torch.matmul(self.rel_send, x)
        edges = torch.cat([senders, receivers], dim=2)
        return edges
    
    def dataloader2MPNN(self, x_dl):
        # Convert DLC dataloader structure to work with MPNN data structure
        x_MPNN = x_dl[:, 0::self.nAgent].view(x_dl.size(0), 1, self.in_dim)
        for i in range(1, self.nAgent):
            x_MPNN = torch.cat([x_MPNN, x_dl[:, i::self.nAgent].view(x_dl.size(0), 1, self.in_dim)], dim=1)
        return x_MPNN
    
    def MPNN2dataloader(self, x_MPNN):
        # Convert MPNN data structure back into DLC dataloader structure
        x_dl = x_MPNN[:, :, 0].view(x_MPNN.size(0), self.nAgent)
        for i in range(1, self.out_dim):
            x_dl = torch.cat([x_dl, x_MPNN[:, :, i].view(x_MPNN.size(0), self.nAgent)], dim=1)
        return x_dl

    def forward(self, inputs):
        x = self.dataloader2MPNN(inputs)
        x = self.MPMLP1(x)
        x = self.node2edge(x)
        x = self.MPMLP2(x)
        x = self.edge2node(x)
        x = self.MPMLP3(x)
        outputs = self.MPNN2dataloader(x)
        return outputs
    
class DataGenerator:
    def __init__(self, ode_func, dist_func, tmax, dt, sampling_freq, onestep_dim, plot_option, aug_option):
        super(DataGenerator, self).__init__()
        self.ode_func = ode_func
        self.dist_func = dist_func
        self.tmax = tmax
        self.dt = dt
        self.sampling_freq = sampling_freq
        self.onestep_dim = onestep_dim
        self.plot_option = plot_option
        self.aug_option = aug_option

    def generate_trajectory(self, gen_set, ode_func_args, dist_func_args):
        # Generate time vector for integration
        t = np.arange(0, self.tmax*self.dt, self.dt)
        # Generate initial condition
        x0 = self.dist_func(dist_func_args[0], dist_func_args[1], (self.onestep_dim,))
        # Integrate trajectory
        x = odeint(self.ode_func, x0, t, args=ode_func_args)
        # Subsample for training
        x_sampled = torch.tensor(x[0::self.sampling_freq], dtype=torch.float32).view(-1, self.onestep_dim*int(self.tmax/self.sampling_freq))
        # Split into input and output
        state_input = x_sampled[:, :self.onestep_dim]
        state_output = x_sampled[:, self.onestep_dim:]
        # Simulate through rest of trajectory
        for _ in tqdm.tqdm(range(gen_set-1)):
            x0 = self.dist_func(dist_func_args[0], dist_func_args[1], (self.onestep_dim,))
            x = odeint(self.ode_func, x0, t, args=ode_func_args)
            x_sampled = torch.tensor(x[0::self.sampling_freq], dtype=torch.float32).view(-1, self.onestep_dim*int(self.tmax/self.sampling_freq))
            state_input = torch.vstack([state_input, x_sampled[:, :self.onestep_dim]])
            state_output = torch.vstack([state_output, x_sampled[:, self.onestep_dim:]])
        # Return
        return state_input, state_output