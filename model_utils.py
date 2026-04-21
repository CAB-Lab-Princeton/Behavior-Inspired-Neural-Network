import torch, os, sys, time, argparse, random, tqdm, modules
import torch.nn.functional as F, matplotlib.pyplot as plt, numpy as np, torch.optim as optim, torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import TensorDataset, DataLoader

def encode_onehot(labels):
    # From NRI paper, one-hot labeling for message passing
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    # Return
    return labels_onehot

def initialise_DLC_model(in_dim, hid_dim, nAgent, nOpinion, nDim, dOrder, device, learning_rate, scheduler_step, scheduler_gamma, arch='DLCMLP', activation='tanh', weight_decay=0.0):
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
    # encoder = torch.nn.DataParallel(encoder, device_ids = [0, 1])
    # NOD_func = torch.nn.DataParallel(NOD_func, device_ids = [0, 1])
    # decoder = torch.nn.DataParallel(decoder, device_ids = [0, 1])
    encoder.to(device)
    NOD_func.to(device)
    decoder.to(device)
    # Initialise optimisation model. AdamW so weight_decay is decoupled (proper
    # weight decay, not L2 on the gradient). Equivalent to Adam when wd=0.
    optimiser = optim.AdamW(
        list(encoder.parameters())+list(NOD_func.parameters())+list(decoder.parameters()),
        lr=learning_rate, weight_decay=weight_decay,
    )
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
    # Produces a_graph of shape (B, nAgent, nAgent) where entry [b, j, i]
    # matches the original implementation exactly:
    #   nDim == 1:        state_t[b, j] - state_t[b, i]          (diag zeroed)
    #   nDim == 2, inv=0: ||p_j - p_i||^2                        (diag naturally 0)
    #   nDim == 2, inv=1: 1 / (||p_j - p_i||^2 + eps)             (diag zeroed)
    if nDim == 1:
        assert nAgent == state_t.size(1)
        # Broadcasted pairwise difference; diagonal is already zero.
        return state_t.unsqueeze(2) - state_t.unsqueeze(1)
    if nDim == 2:
        x = state_t[:, :nAgent]
        y = state_t[:, nAgent:2*nAgent]
        dx = x.unsqueeze(2) - x.unsqueeze(1)
        dy = y.unsqueeze(2) - y.unsqueeze(1)
        Dist2 = dx * dx + dy * dy
        if inverse:
            epsilon = 1E-3
            a_graph = 1.0 / (Dist2 + epsilon)
            # Zero the diagonal (which would otherwise be 1/eps). Mask is tiny
            # so we build it inline rather than caching — caching a module-level
            # tensor breaks torch.compile mode='reduce-overhead' (CUDA graphs).
            a_graph = a_graph * (1.0 - torch.eye(nAgent, device=a_graph.device, dtype=a_graph.dtype))
            return a_graph
        # Non-inverse: diagonal is naturally 0.
        return Dist2
    
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
    # Original inner sum:
    #   alpha*z + A@z + z@Oᵀ + (A@z)@Oᵀ
    # Using (A@z)@Oᵀ = A@(z@Oᵀ) and distributivity:
    #   = alpha*z + z@Oᵀ + A@(z + z@Oᵀ)
    # One matmul (z, oT) + one bmm (a, .) instead of two bmm's and two matmul's.
    zo = torch.matmul(z_t_dlc, o_graph.t())                         # (B, N, No)
    inner = alpha * z_t_dlc + zo + torch.matmul(a_graph, z_t_dlc + zo)
    dz_t_dlc = -d * z_t_dlc + torch.tanh(u * inner) + b
    # Update decision
    z_tp1_dlc = z_t_dlc + dt * dz_t_dlc
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

def _DLC_unroll_impl(encoder, NOD_func, decoder, state_input, data_offset,
                     nAgent, nOpinion, nDim, dOrder, batch_size, dt, device, inverse):
    """Full RNN forward unroll: encoder -> (NOD + decoder)*data_offset.
    Returns (state_tpn_dlc, z_t_dlc, z_tp1_dlc) — same triple the in-line loop
    produced. Exposed as a standalone function so torch.compile can see the
    full loop and fuse across iterations.
    """
    state_tp1, z_t, z_tp1 = DLC_step(
        encoder, NOD_func, decoder,
        nAgent, nOpinion, nDim, dOrder, batch_size,
        state_input, dt, device, inverse,
    )
    state_list = [state_tp1]
    tmp = state_tp1
    z_tmp = z_tp1
    for _ in range(data_offset - 1):
        tmp, z_tmp = DLC_RNN_step(
            z_tmp, NOD_func, decoder,
            nAgent, nOpinion, nDim, dOrder, batch_size,
            tmp, dt, device, inverse,
        )
        state_list.append(tmp)
    state_tpn = torch.cat(state_list, dim=1) if len(state_list) > 1 else state_list[0]
    return state_tpn, z_t, z_tp1


# Module-level name that train/valid/test call. `enable_compile` rebinds this
# to a torch.compile-wrapped version when requested.
DLC_unroll = _DLC_unroll_impl

# Whether CUDA graphs (via torch.compile mode='reduce-overhead') are active.
# When True, train/valid loops call torch.compiler.cudagraph_mark_step_begin()
# before each batch to avoid "output overwritten by next graph replay" errors.
_CUDAGRAPH_ACTIVE = False


def _step_mark():
    """Signal the boundary between batches when CUDA graphs are active."""
    if _CUDAGRAPH_ACTIVE:
        torch.compiler.cudagraph_mark_step_begin()


def enable_compile(mode="default"):
    """Opt-in: wrap DLC_unroll with torch.compile. Call once after the model is
    initialised. `mode`:
      - 'default'          — ~2x speedup on fixed shapes, robust (recommended).
      - 'reduce-overhead'  — uses CUDA graphs; ~5-20% additional speedup but
                             can interact badly with module-level cached tensors
                             and requires mark_step_begin() between batches
                             (handled here). Prefer 'default' for stability.
      - 'max-autotune'     — longer warmup, uses autotuned kernels.
    """
    global DLC_unroll, _CUDAGRAPH_ACTIVE
    DLC_unroll = torch.compile(_DLC_unroll_impl, mode=mode)
    _CUDAGRAPH_ACTIVE = (mode == "reduce-overhead")
    print(f"[model_utils] torch.compile enabled with mode='{mode}'")


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
        _step_mark()
        # Extract input and output data
        state_input_train_batch, state_output_train_batch = data_train
        # Send batch data to CUDA
        state_input_train_batch, state_output_train_batch = state_input_train_batch.to(device), state_output_train_batch.to(device)
        # Full RNN unroll (torch.compile-friendly single entry point).
        state_tpn_dlc, z_t_dlc, z_tp1_dlc = DLC_unroll(
            encoder, NOD_func, decoder, state_input_train_batch, data_offset,
            nAgent, nOpinion, nDim, dOrder, batch_size, dt, device, inverse,
        )
        # Training loss
        loss, loss_pred, loss_recon, loss_latent = loss_function(encoder, decoder, state_tpn_dlc, z_t_dlc, z_tp1_dlc, state_input_train_batch, state_output_train_batch, batch_size, nAgent, nOpinion, nDim, dOrder)
        # Update model
        optimiser.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(NOD_func.parameters()) + list(decoder.parameters()),
            max_norm=1.0,
        )
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
            _step_mark()
            # Extract input and output data
            state_input_valid_batch, state_output_valid_batch = data_valid
            # Send batch data to CUDA
            state_input_valid_batch, state_output_valid_batch = state_input_valid_batch.to(device), state_output_valid_batch.to(device)
            # Full RNN unroll (torch.compile-friendly single entry point).
            state_tpn_dlc, z_t_dlc, z_tp1_dlc = DLC_unroll(
                encoder, NOD_func, decoder, state_input_valid_batch, data_offset,
                nAgent, nOpinion, nDim, dOrder, batch_size, dt, device, inverse,
            )
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
    # assert len(test_loader) == 1
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
        state_tp1_dlc, z_t_dlc, z_tp1_dlc = DLC_step(encoder, NOD_func, decoder, nAgent, nOpinion, nDim, dOrder, batch_size, state_input_test_batch, dt, device, inverse)
        # Loop for RNN testing: accumulate into lists, cat once at the end.
        state_list = [state_tp1_dlc]
        b_list = [NOD_func.b(state_input_test_batch)]
        z_list = [torch.reshape(z_tp1_dlc, (batch_size, nAgent*nOpinion))]
        state_tpn_temp = state_tp1_dlc
        z_t_temp = z_tp1_dlc
        for _ in range(data_offset-1):
            b_list.append(NOD_func.b(state_tpn_temp))
            state_tpn_temp, z_t_temp = DLC_RNN_step(z_t_temp, NOD_func, decoder, nAgent, nOpinion, nDim, dOrder, batch_size, state_tpn_temp, dt, device, inverse)
            state_list.append(state_tpn_temp)
            z_list.append(torch.reshape(z_t_temp, (batch_size, nAgent*nOpinion)))
        state_tpn_dlc = torch.cat(state_list, dim=1) if len(state_list) > 1 else state_list[0]
        b_tpn_dlc = torch.cat(b_list, dim=1) if len(b_list) > 1 else b_list[0]
        z_tpn_dlc_return = torch.cat(z_list, dim=1) if len(z_list) > 1 else z_list[0]
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