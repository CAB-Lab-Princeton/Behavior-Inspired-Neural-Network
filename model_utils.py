import torch, os, sys, time, argparse, random, tqdm, modules
import torch.nn.functional as F, matplotlib.pyplot as plt, numpy as np, torch.optim as optim, torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import TensorDataset, DataLoader

def encode_onehot(labels):
    # From NRI, one-hot labeling for message passing
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    # Return
    return labels_onehot

def initialise_DLC_model(in_dim, hid_dim, nAgent, nOpinion, nDim, dOrder, device, learning_rate, scheduler_step, scheduler_gamma, arch='DLCMLP', activation='tanh'):
    # Initialise encoder
    if arch == 'DLCMLP':
        encoder = modules.DLCMLP(in_dim=in_dim, hid_dim=hid_dim, out_dim=nAgent*nOpinion)
    elif arch == 'MPNN':
        off_diag = np.ones([nAgent, nAgent]) - np.eye(nAgent)
        rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
        rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
        rel_rec = torch.FloatTensor(rel_rec).to(device)
        rel_send = torch.FloatTensor(rel_send).to(device)
        encoder = modules.MPNN(nAgent=nAgent, in_dim=nDim*dOrder, hid_dim=hid_dim, out_dim=nOpinion, rel_rec=rel_rec, rel_send=rel_send, activation=activation)
    # Initialise nonlinear opinion dynamics
    NOD_func = modules.NonlinearOpinionDynamics(nAgent=nAgent, nOpinion=nOpinion, nDim=nDim, dOrder=dOrder, in_dim=in_dim, hid_dim=hid_dim, device=device, arch=arch)
    # Initialise decoder
    if dOrder == 1:
        if arch =='DLCMLP':
            decoder = modules.DLCMLP(in_dim=nAgent*nOpinion, hid_dim=hid_dim, out_dim=in_dim)
        elif arch == 'MPNN':
            off_diag = np.ones([nAgent, nAgent]) - np.eye(nAgent)
            rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
            rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
            rel_rec = torch.FloatTensor(rel_rec).to(device)
            rel_send = torch.FloatTensor(rel_send).to(device)
            decoder = modules.MPNN(nAgent=nAgent, in_dim=nOpinion, hid_dim=hid_dim, out_dim=nDim*dOrder, rel_rec=rel_rec, rel_send=rel_send, activation=activation)
    elif dOrder == 2:
        if arch =='DLCMLP':
            decoder = modules.DLCMLP(in_dim=nAgent*nOpinion, hid_dim=hid_dim, out_dim=int(in_dim/2))
        elif arch == 'MPNN':
            off_diag = np.ones([nAgent, nAgent]) - np.eye(nAgent)
            rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
            rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
            rel_rec = torch.FloatTensor(rel_rec).to(device)
            rel_send = torch.FloatTensor(rel_send).to(device)
            decoder = modules.MPNN(nAgent=nAgent, in_dim=nOpinion, hid_dim=hid_dim, out_dim=int(nDim*dOrder/2), rel_rec=rel_rec, rel_send=rel_send, activation=activation)
    # Send DLC modules to training device
    encoder.to(device)
    NOD_func.to(device)
    decoder.to(device)
    # Initialise optimisation model
    optimiser = optim.Adam(list(encoder.parameters())+list(NOD_func.parameters())+list(decoder.parameters()), lr=learning_rate)
    # Initialise optimisation scheduler
    scheduler = lr_scheduler.StepLR(optimiser, step_size=scheduler_step, gamma=scheduler_gamma)
    # Return
    return encoder, NOD_func, decoder, optimiser, scheduler

def save_DLC_model(loss_pred_valid, loss_pred_valid_lowest, valid_loader, loss_pred_train, loss_pred_train_lowest, train_loader, encoder, NOD_func, decoder, timestr):
    if loss_pred_valid/len(valid_loader) < loss_pred_valid_lowest:
        # Set new lowerest validation prediction lost
        print(loss_pred_valid/len(valid_loader), loss_pred_valid_lowest)
        loss_pred_valid_lowest = loss_pred_valid/len(valid_loader)
        # Set DLC saving pathway
        if timestr == "":
            savepath = 'Model/'
        else:
            savepath = 'Model/' + timestr + '/'
        # Save new trained model
        torch.save(encoder.state_dict(), savepath+'encoder_model.pt')
        torch.save(NOD_func.state_dict(), savepath+'NOD_func_model.pt')
        torch.save(decoder.state_dict(), savepath+'decoder_model.pt')
    if loss_pred_train/len(train_loader) < loss_pred_train_lowest:
        loss_pred_train_lowest = loss_pred_train/len(train_loader)
    return loss_pred_valid_lowest, loss_pred_train_lowest

def load_DLC_model(in_dim, hid_dim, nAgent, nOpinion, nDim, dOrder, device, timestr, arch='DLCMLP', activation='tanh'):
    # Initialise encoder
    if arch == 'DLCMLP':
        encoder = modules.DLCMLP(in_dim=in_dim, hid_dim=hid_dim, out_dim=nAgent*nOpinion)
    elif arch == 'MPNN':
        off_diag = np.ones([nAgent, nAgent]) - np.eye(nAgent)
        rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
        rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
        rel_rec = torch.FloatTensor(rel_rec).to(device)
        rel_send = torch.FloatTensor(rel_send).to(device)
        encoder = modules.MPNN(nAgent=nAgent, in_dim=nDim*dOrder, hid_dim=hid_dim, out_dim=nOpinion, rel_rec=rel_rec, rel_send=rel_send, activation=activation)
    # Initialise nonlinear opinion dynamics
    NOD_func = modules.NonlinearOpinionDynamics(nAgent=nAgent, nOpinion=nOpinion, nDim=nDim, dOrder=dOrder, in_dim=in_dim, hid_dim=hid_dim, device=device, arch=arch)
    # Initialise decoder
    if dOrder == 1:
        if arch =='DLCMLP':
            decoder = modules.DLCMLP(in_dim=nAgent*nOpinion, hid_dim=hid_dim, out_dim=in_dim)
        elif arch == 'MPNN':
            off_diag = np.ones([nAgent, nAgent]) - np.eye(nAgent)
            rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
            rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
            rel_rec = torch.FloatTensor(rel_rec).to(device)
            rel_send = torch.FloatTensor(rel_send).to(device)
            decoder = modules.MPNN(nAgent=nAgent, in_dim=nOpinion, hid_dim=hid_dim, out_dim=nDim*dOrder, rel_rec=rel_rec, rel_send=rel_send, activation=activation)
    elif dOrder == 2:
        if arch =='DLCMLP':
            decoder = modules.DLCMLP(in_dim=nAgent*nOpinion, hid_dim=hid_dim, out_dim=int(in_dim/2))
        elif arch == 'MPNN':
            off_diag = np.ones([nAgent, nAgent]) - np.eye(nAgent)
            rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
            rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
            rel_rec = torch.FloatTensor(rel_rec).to(device)
            rel_send = torch.FloatTensor(rel_send).to(device)
            decoder = modules.MPNN(nAgent=nAgent, in_dim=nOpinion, hid_dim=hid_dim, out_dim=int(nDim*dOrder/2), rel_rec=rel_rec, rel_send=rel_send, activation=activation)
    # Set DLC loading pathway
    if timestr == "":
        savepath = 'Model/'
    else:
        savepath = 'Model/' + timestr + '/'
    # Load DLC module parameters
    encoder.load_state_dict(torch.load(savepath+'encoder_model.pt'))
    NOD_func.load_state_dict(torch.load(savepath+'NOD_func_model.pt'))
    decoder.load_state_dict(torch.load(savepath+'decoder_model.pt'))
    # Send DLC modules to training device
    encoder.to(device)
    NOD_func.to(device)
    decoder.to(device)
    # Return
    return encoder, NOD_func, decoder

def calc_aGraph(state_t, NOD_func, nAgent, device, nDim, inverse):
    if nDim == 1:
        assert nAgent == len(state_t[0])
        a_graph = ((state_t[:, :]-state_t[:, 0].unsqueeze(dim=1))).unsqueeze(dim=2)
        for i in range(len(state_t[0])-1):
            a_graph = torch.cat([a_graph, ((state_t[:, :]-state_t[:, i+1].unsqueeze(dim=1))).unsqueeze(dim=2)], dim=2)
        for i in range(nAgent):
            a_graph[:, i, i] = 0
        return a_graph
    if nDim == 2:
        if inverse:
            epsilon = 1E-3
            xDist = state_t[:, :nAgent]-state_t[:, 0].unsqueeze(dim=1)
            yDist = state_t[:, nAgent:]-state_t[:, nAgent].unsqueeze(dim=1)
            Dist = xDist**2 + yDist**2
            a_graph = (1.0/(Dist+epsilon)).unsqueeze(dim=2)
            for i in range(nAgent-1):
                xDist = state_t[:, :nAgent]-state_t[:, i+1].unsqueeze(dim=1)
                yDist = state_t[:, nAgent:]-state_t[:, i+1+nAgent].unsqueeze(dim=1)
                Dist = xDist**2 + yDist**2
                a_graph = torch.cat([a_graph, (1.0/(Dist+epsilon)).unsqueeze(dim=2)], dim=2)
            for i in range(nAgent):
                a_graph[:, i, i] = 0
            # a_graph = torch.mul(NOD_func.aGraph_prefactor, a_graph)
            return a_graph
        else:
            xDist = state_t[:, :nAgent]-state_t[:, 0].unsqueeze(dim=1)
            yDist = state_t[:, nAgent:]-state_t[:, nAgent].unsqueeze(dim=1)
            Dist = xDist**2 + yDist**2
            a_graph = Dist.unsqueeze(dim=2)
            for i in range(nAgent-1):
                xDist = state_t[:, :nAgent]-state_t[:, i+1].unsqueeze(dim=1)
                yDist = state_t[:, nAgent:]-state_t[:, i+1+nAgent].unsqueeze(dim=1)
                Dist = xDist**2 + yDist**2
                a_graph = torch.cat([a_graph, Dist.unsqueeze(dim=2)], dim=2)
            # a_graph = torch.mul(NOD_func.aGraph_prefactor, a_graph)
            return a_graph
    
def calc_oGraph(nOpinion, NOD_func, device):
    o_graph = torch.zeros(nOpinion, nOpinion).to(device)
    o_graph[NOD_func.oGraph_upper_index[0], NOD_func.oGraph_upper_index[1]] = NOD_func.oGraph_upper
    o_graph[NOD_func.oGraph_lower_index[0], NOD_func.oGraph_lower_index[1]] = NOD_func.oGraph_lower
    return o_graph

def calc_NOD(state_t, z_t_dlc, NOD_func, batch_size, nAgent, nOpinion, nDim, dt, device, inverse):
    # Split parameters
    d = NOD_func.d**2
    u = NOD_func.u**2
    alpha = NOD_func.alpha**2
    b = torch.reshape(NOD_func.b(state_t), (batch_size, nAgent, nOpinion))
    a_graph = calc_aGraph(state_t[:, :nDim*nAgent], NOD_func, nAgent, device, nDim, inverse)
    o_graph = calc_oGraph(nOpinion, NOD_func, device)
    # Update decision dot
    dz_t_dlc = -d*z_t_dlc + torch.tanh(u*(alpha*z_t_dlc + torch.matmul(a_graph, z_t_dlc) + torch.matmul(z_t_dlc, o_graph.t()) + torch.matmul(torch.matmul(a_graph, z_t_dlc), o_graph.t()))) + b
    # Update decision
    z_tp1_dlc = z_t_dlc + dt*dz_t_dlc
    # Return
    return z_tp1_dlc

def DLC_step(encoder, NOD_func, decoder, nAgent, nOpinion, nDim, dOrder, batch_size, state_t, dt, device, inverse):
    # Encode latent decision
    z_t_dlc = encoder(state_t)
    # Reshape encoded latent deicison
    z_t_dlc = torch.reshape(z_t_dlc, (batch_size, nAgent, nOpinion))
    # Encoder model parameters
    z_tp1_dlc = calc_NOD(state_t, z_t_dlc, NOD_func, batch_size, nAgent, nOpinion, nDim, dt, device, inverse)
    # Update state
    if dOrder == 1:
        state_tp1_dlc = decoder(torch.reshape(z_tp1_dlc, (batch_size, nAgent*nOpinion)))
    elif dOrder == 2:
        pos_tp1_dlc = state_t[:, :nDim*nAgent] + dt*state_t[:, nDim*nAgent:]
        vel_tp1_dlc = decoder(torch.reshape(z_tp1_dlc, (batch_size, nAgent*nOpinion)))
        state_tp1_dlc = torch.hstack([pos_tp1_dlc, vel_tp1_dlc])
    # Return
    return state_tp1_dlc, z_t_dlc, z_tp1_dlc

def DLC_RNN_step(z_t_dlc, NOD_func, decoder, nAgent, nOpinion, nDim, dOrder, batch_size, state_t, dt, device, inverse):
    # Encoder model parameters
    z_tp1_dlc = calc_NOD(state_t, z_t_dlc, NOD_func, batch_size, nAgent, nOpinion, nDim, dt, device, inverse)
    # Update state
    if dOrder == 1:
        state_tp1_dlc = decoder(torch.reshape(z_tp1_dlc, (batch_size, nAgent*nOpinion)))
    elif dOrder == 2:
        pos_tp1_dlc = state_t[:, :nDim*nAgent] + dt*state_t[:, nDim*nAgent:]
        vel_tp1_dlc = decoder(torch.reshape(z_tp1_dlc, (batch_size, nAgent*nOpinion)))
        state_tp1_dlc = torch.hstack([pos_tp1_dlc, vel_tp1_dlc])
    # Return
    return state_tp1_dlc, z_tp1_dlc

def loss_function(encoder, decoder, state_tpn_dlc, z_t_dlc, z_tp1_dlc, state_input_batch, state_output_batch, batch_size, nAgent, nOpinion, nDim, dOrder):
    # Prediction loss
    loss_pred = F.mse_loss(state_tpn_dlc, state_output_batch)
    # Reconstruction loss
    if dOrder == 1:
        state_t_dlc_recon = decoder(torch.reshape(z_t_dlc, (batch_size, nAgent*nOpinion)))
    elif dOrder == 2:
        v_t_dlc_recon = decoder(torch.reshape(z_t_dlc, (batch_size, nAgent*nOpinion)))
        state_t_dlc_recon = torch.hstack([state_input_batch[:, :nDim*nAgent], v_t_dlc_recon])
    loss_recon = F.mse_loss(state_t_dlc_recon, state_input_batch)
    # Latent loss
    z_tp1_dlc_latent = encoder(state_output_batch[:, :dOrder*nDim*nAgent])
    z_tp1_dlc_latent = torch.reshape(z_tp1_dlc_latent, (batch_size, nAgent, nOpinion))
    loss_latent = F.mse_loss(z_tp1_dlc, z_tp1_dlc_latent)
    # Training loss
    loss = loss_pred + loss_recon + loss_latent
    # Return
    return loss, loss_pred, loss_recon, loss_latent

def train_epoch(encoder, NOD_func, decoder, optimiser, scheduler, device, train_loader, data_offset, nAgent, nOpinion, nDim, dOrder, batch_size, dt, inverse):
    # Initialise loop loss
    loss_train, loss_pred_train, loss_recon_train, loss_latent_train = 0, 0, 0, 0
    # Set models to training
    encoder.train()
    NOD_func.train()
    decoder.train()
    # Run training loop
    for n, data_train in enumerate(train_loader):
        # Extract input and output data
        state_input_train_batch, state_output_train_batch = data_train
        # Send batch data to CUDA
        state_input_train_batch, state_output_train_batch = state_input_train_batch.to(device), state_output_train_batch.to(device)
        # Intial training step
        state_tp1_dlc, z_t_dlc, z_tp1_dlc = DLC_step(encoder, NOD_func, decoder, nAgent, nOpinion, nDim, dOrder, batch_size, state_input_train_batch, dt, device, inverse)
        # Loop for RNN training
        state_tpn_dlc = state_tp1_dlc
        state_tpn_temp = state_tp1_dlc
        z_t_temp = z_tp1_dlc
        for _ in range(data_offset-1):
            state_tpn_temp, z_t_temp = DLC_RNN_step(z_t_temp, NOD_func, decoder, nAgent, nOpinion, nDim, dOrder, batch_size, state_tpn_temp, dt, device, inverse)
            state_tpn_dlc = torch.hstack([state_tpn_dlc, state_tpn_temp])
        # Training loss
        loss, loss_pred, loss_recon, loss_latent = loss_function(encoder, decoder, state_tpn_dlc, z_t_dlc, z_tp1_dlc, state_input_train_batch, state_output_train_batch, batch_size, nAgent, nOpinion, nDim, dOrder)
        # Update model
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        # Update loss values
        loss_train += loss.item()
        loss_pred_train += loss_pred.item()
        loss_recon_train += loss_recon.item()
        loss_latent_train += loss_latent.item()
    # Update scheduler
    scheduler.step()
    # Return
    return loss_train, loss_pred_train, loss_recon_train, loss_latent_train

def validate_epoch(encoder, NOD_func, decoder, device, valid_loader, data_offset, nAgent, nOpinion, nDim, dOrder, batch_size, dt, inverse):
    # Initialise loop loss
    loss_valid, loss_pred_valid, loss_recon_valid, loss_latent_valid = 0, 0, 0, 0
    # Run validation loop
    with torch.no_grad():
        for m, data_valid in enumerate(valid_loader):
            # Extract input and output data
            state_input_valid_batch, state_output_valid_batch = data_valid
            # Send batch data to CUDA
            state_input_valid_batch, state_output_valid_batch = state_input_valid_batch.to(device), state_output_valid_batch.to(device)
            # Intial training step
            state_tp1_dlc, z_t_dlc, z_tp1_dlc = DLC_step(encoder, NOD_func, decoder, nAgent, nOpinion, nDim, dOrder, batch_size, state_input_valid_batch, dt, device, inverse)
            # Loop for RNN training
            state_tpn_dlc = state_tp1_dlc
            state_tpn_temp = state_tp1_dlc
            z_t_temp = z_tp1_dlc
            for _ in range(data_offset-1):
                state_tpn_temp, z_t_temp = DLC_RNN_step(z_t_temp, NOD_func, decoder, nAgent, nOpinion, nDim, dOrder, batch_size, state_tpn_temp, dt, device, inverse)
                state_tpn_dlc = torch.hstack([state_tpn_dlc, state_tpn_temp])
            # Training loss
            loss, loss_pred, loss_recon, loss_latent = loss_function(encoder, decoder, state_tpn_dlc, z_t_dlc, z_tp1_dlc, state_input_valid_batch, state_output_valid_batch, batch_size, nAgent, nOpinion, nDim, dOrder)
            # Update loss values
            loss_valid += loss.item()
            loss_pred_valid += loss_pred.item()
            loss_recon_valid += loss_recon.item()
            loss_latent_valid += loss_latent.item()
    # Return
    return loss_valid, loss_pred_valid, loss_recon_valid, loss_latent_valid

def test_epoch(encoder, NOD_func, decoder, device, test_loader, data_offset, nAgent, nOpinion, nDim, dOrder, dt, inverse):
    # Set model to evaluation
    encoder.eval()
    NOD_func.eval()
    decoder.eval()
    # Check testing time bath length
    assert len(test_loader) == 1
    # Initialise loop loss
    loss_test, loss_pred_test, loss_recon_test, loss_latent_test = 0, 0, 0, 0
    # Run testing loop
    for m, data_test in enumerate(test_loader):
        # Extract input and output data
        state_input_test_batch, state_output_test_batch = data_test
        batch_size = len(state_input_test_batch)
        # Send batch data to CUDA
        state_input_test_batch, state_output_test_batch = state_input_test_batch.to(device), state_output_test_batch.to(device)
        # Intial training step
        b_tpn_dlc = NOD_func.b(state_input_test_batch)
        state_tp1_dlc, z_t_dlc, z_tp1_dlc = DLC_step(encoder, NOD_func, decoder, nAgent, nOpinion, nDim, dOrder, batch_size, state_input_test_batch, dt, device, inverse)
        # Loop for RNN training
        state_tpn_dlc = state_tp1_dlc
        state_tpn_temp = state_tp1_dlc
        z_tpn_dlc = z_tp1_dlc
        z_tpn_dlc_return = torch.reshape(z_tp1_dlc, (batch_size, nAgent*nOpinion))
        z_t_temp = z_tp1_dlc
        for _ in range(data_offset-1):
            b_tpn_dlc = torch.hstack([b_tpn_dlc, NOD_func.b(state_tpn_temp)])
            state_tpn_temp, z_t_temp = DLC_RNN_step(z_t_temp, NOD_func, decoder, nAgent, nOpinion, nDim, dOrder, batch_size, state_tpn_temp, dt, device, inverse)
            state_tpn_dlc = torch.hstack([state_tpn_dlc, state_tpn_temp])
            z_tpn_dlc_return = torch.hstack([z_tpn_dlc_return, torch.reshape(z_t_temp, (batch_size, nAgent*nOpinion))])
        # Training loss
        loss, loss_pred, loss_recon, loss_latent = loss_function(encoder, decoder, state_tpn_dlc, z_t_dlc, z_tp1_dlc, state_input_test_batch, state_output_test_batch, batch_size, nAgent, nOpinion, nDim, dOrder)
        # Update loss values
        loss_test += loss.item()
        loss_pred_test += loss_pred.item()
        loss_recon_test += loss_recon.item()
        loss_latent_test += loss_latent.item()
    # Print testing loss results
    print("Testing Loss: ", loss/len(test_loader), ", Test Prediction Loss: ", loss_pred/len(test_loader), ", Test Reconstruction Loss: ", loss_recon/len(test_loader))
    # Shortened loss
    print(state_tpn_dlc.size())
    print(F.mse_loss(state_tpn_dlc[:, 0:200], state_output_test_batch[:, 0:200]))
    # Return
    return loss_test, loss_pred_test, loss_recon_test, loss_latent_test, state_tpn_dlc, z_tpn_dlc_return, b_tpn_dlc