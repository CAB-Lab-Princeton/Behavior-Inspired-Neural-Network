import torch, sys, mpld3, os
import numpy as np, matplotlib.pyplot as plt
import model_utils
from matplotlib.lines import Line2D

def lighten_color(color, amount=0.5):
    """
    From https://stackoverflow.com/questions/37765197/darken-or-lighten-a-color-in-matplotlib
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

def write_sh(args, timestr):
    # Save loss plot
    savepath = 'Model/' + timestr + '/run.sh'
    # Create savepath
    if not os.path.exists('Model/' + timestr):
        os.makedirs('Model/' + timestr)
    # Write run parameters to model folder
    with open (savepath, 'w') as file:
        file.write("#!/bin/sh\n")
        file.write("python3 Run.py --train=1 --test=1")
        file.write(" --arch=" + str(args["arch"]))
        file.write(" --num_worker=" + str(args["num_worker"]))
        file.write(" --nOpinion=" + str(args["nOpinion"]))
        file.write(" --hid_dim=" + str(args["hid_dim"]))
        file.write(" --epoch=" + str(args["epoch"]))
        file.write(" --learning_rate=" + str(args["learning_rate"]))
        file.write(" --batch_size=" + str(args["batch_size"]))
        file.write(" --scheduler_step=" + str(args["scheduler_step"]))
        file.write(" --scheduler_gamma=" + str(args["scheduler_gamma"]))
        file.write(" --activation=" + str(args["activation"]))
        file.write(" --train_set=" + str(args["train_set"]))
        file.write(" --dataset=" + str(args["dataset"]))
        file.write(" --seed=" + str(args["seed"]))
        file.write(" --inverse=" + str(args["inverse"]))
        file.write(" --dOrder=" + str(args["dOrder"]))

def print_DLC_model_parameter(encoder, NOD_function, decoder):
    for name, param in encoder.named_parameters():
        if param.requires_grad:
            print (name, param.data)
    for name, param in NOD_function.named_parameters():
        if param.requires_grad:
            print (name, param.data)
    for name, param in decoder.named_parameters():
        if param.requires_grad:
            print (name, param.data)

def print_NOD_parameters(NOD_function, nOpinion, nAgent, device, timestr):
    # Set file name
    name = "NOD_Parameter.txt"
    # Save loss plot
    savepath = 'Model/' + timestr + '/' + name
    # Calculate o_graph
    o_graph = model_utils.calc_oGraph(nOpinion, NOD_func=NOD_function, device=device)
    # Write lowest loss values to file
    with open (savepath, 'w') as file: 
        file.write("NOD Function u: \n") 
        file.write(str(NOD_function.u**2)+"\n")  
        file.write("NOD Function d: \n") 
        file.write(str(NOD_function.d**2)+"\n")  
        file.write("NOD Function alpha: \n") 
        file.write(str(NOD_function.alpha**2)+"\n")  
        file.write("NOD Function o_graph: \n") 
        file.write(str(o_graph)+"\n") 

def append_loss(loss, loss_pred, loss_recon, loss_latent, loss_log, loss_pred_log, loss_recon_log, loss_latent_log, dataloader, name):
    # Append losses to lists
    loss_log.append(loss/len(dataloader))
    loss_pred_log.append(loss_pred/len(dataloader))
    loss_recon_log.append(loss_recon/len(dataloader))
    loss_latent_log.append(loss_latent/len(dataloader))
    # Print new epoch updated losses
    print(name, " Loss: ", loss/len(dataloader), ", ", name, " Prediction Loss: ", loss_pred/len(dataloader), ", ", name, " Reconstruction Loss: ", loss_recon/len(dataloader))

def print_lowest_loss(loss_pred_valid_lowest, loss_pred_train_lowest, timestr):
    # Set file name
    name = "Lowest_Loss.txt"
    # Save loss plot
    savepath = 'Model/' + timestr + '/' + name
    # Wriet lowest loss values to file
    with open (savepath, 'w') as file: 
        file.write("Lowest Training Loss: \n") 
        file.write(str(loss_pred_train_lowest)+"\n")  
        file.write("Lowest Validation Loss: \n") 
        file.write(str(loss_pred_valid_lowest)+"\n")  

def plot_loss_figure(epoch, train_loss_log, train_pred_loss_log, train_recon_loss_log, train_latent_loss_log, timestr, name):
    # Initialise plot
    fig = plt.figure(figsize=(6, 6), dpi=100)
    # Initialise epoch for plotting
    loss_plot = np.linspace(1, epoch, len(train_pred_loss_log))
    # Plot loss
    plt.semilogy(loss_plot, train_loss_log)
    plt.semilogy(loss_plot, train_pred_loss_log)
    plt.semilogy(loss_plot, train_recon_loss_log)
    plt.semilogy(loss_plot, train_latent_loss_log)
    plt.legend(["Training", "Prediction",  "Reconstruction",  "Latent"])
    plt.title(name + " Loss Plots")
    plt.xlabel("$t$")
    plt.ylabel(name + " Loss")
    plt.savefig('Model/' + timestr + '/' + name + '_loss_plot.png')
    # Save loss plot
    savepath = 'Model/' + timestr + '/' + name + '_loss_plot.html'
    html_str = mpld3.fig_to_html(fig)
    Html_file= open(savepath, "w")
    Html_file.write(html_str)
    Html_file.close()

def plot_test_pendulum(test_loader, state_tpn_dlc, z_tpn_dlc, b_tpn_dlc, dt, selected_trajectory, timestr):
    # Extract ground truth data
    _, state_output_test_batch = next(iter(test_loader))
    # Initialise plot
    fig = plt.figure(figsize=(13, 3), dpi=300)
    # Initialise time for plotting
    time = np.linspace(1, len(state_output_test_batch[0, 0::2]), len(state_output_test_batch[0, 0::2]))
    # Plot state space prediction
    plt.subplot(1, 2, 1)
    plt.plot(time, state_tpn_dlc[selected_trajectory, 0::2].cpu().detach().numpy(), color="C0")
    plt.plot(time, state_output_test_batch[selected_trajectory, 0::2].cpu().detach().numpy(), linestyle='dashed', color="C0")
    plt.plot(time, state_tpn_dlc[selected_trajectory, 1::2].cpu().detach().numpy(), color="C1")
    plt.plot(time, state_output_test_batch[selected_trajectory, 1::2].cpu().detach().numpy(), linestyle='dashed', color="C1")
    plt.legend(["NN Theta", "GT Theta", "NN Theta Dot", "GT Theta Dot"])
    plt.title("Predicted State Space")
    plt.ylabel("$\\theta_{t}$/$\dot{\\theta}_{t}$")
    plt.xlabel("$t$")
    # Plot latent space prediction
    plt.subplot(1, 2, 2)
    plt.plot(time, z_tpn_dlc[selected_trajectory, 0::2].cpu().detach().numpy())
    plt.plot(time, z_tpn_dlc[selected_trajectory, 1::2].cpu().detach().numpy())
    plt.title("Predicted Latent Space")
    plt.legend(["$\hat{z}_{1}$", "$\hat{z}_{2}$"])
    plt.ylabel("$z$")
    plt.xlabel("$t$")
    # Save testing plot PNG
    savepath = 'Model/' + timestr + "/Testing.png"
    plt.subplots_adjust(bottom=0.2)
    plt.savefig(savepath)
    # Save testing plot HTML
    savepath = 'Model/' + timestr + "/Testing.html"
    html_str = mpld3.fig_to_html(fig)
    Html_file= open(savepath, "w")
    Html_file.write(html_str)
    Html_file.close()

def plot_test_doublependulum(test_loader, state_tpn_dlc, z_tpn_dlc, b_tpn_dlc, dt, selected_trajectory, timestr):
    # Set font size
    fz = 15.0
    # Extract ground truth data
    _, state_output_test_batch = next(iter(test_loader))
    # Initialise plot
    fig = plt.figure(figsize=(8, 12), dpi=300)
    # Initialise time for plotting
    time = np.linspace(1, len(state_output_test_batch[0, 0::4]), len(state_output_test_batch[0, 0::4]))
    # Plot state space prediction
    plt.subplot(4, 1, 1)
    # plt.plot(time, state_tpn_dlc[selected_trajectory, 0::4].cpu().detach().numpy(), color="C0")
    # plt.plot(time, state_tpn_dlc[selected_trajectory, 2::4].cpu().detach().numpy(), color="C1")
    # plt.plot(time, state_output_test_batch[selected_trajectory, 0::4].cpu().detach().numpy(), linestyle='dashed', color="C0")
    # plt.plot(time, state_output_test_batch[selected_trajectory, 2::4].cpu().detach().numpy(), linestyle='dashed', color="C1")
    plt.plot(time, state_output_test_batch[selected_trajectory, 0::4].cpu().detach().numpy(), color="C0", linewidth=3)
    plt.plot(time, state_output_test_batch[selected_trajectory, 2::4].cpu().detach().numpy(), color="C1", linewidth=3)
    plt.axhline(0.0, linestyle='dashed', color="black")
    plt.axvspan(0.0, 11.75, color='grey', alpha=0.2)
    plt.axvspan(11.75, 25.5, color=lighten_color('grey', 2.0), alpha=0.2)
    plt.axvspan(25.5, 37.0, color='grey', alpha=0.2)
    plt.axvspan(37.0, 45.0, color=lighten_color('grey', 2.0), alpha=0.2)
    plt.axvspan(45.0, time[-1], color='grey', alpha=0.2)
            # plt.axvline(12.0, linestyle='dashed', color="C2")
            # plt.axvline(25.5, linestyle='dashed', color="C3")
            # plt.axvline(37.0, linestyle='dashed', color="C4")
            # plt.axvline(45.0, linestyle='dashed', color="C5")
    plt.title("Bob 1 Observation Space", fontsize=fz)
    plt.legend(["$\\theta_{1,t}$", "$\dot{\\theta}_{1,t}$"], fontsize=fz)
    plt.xticks(fontsize=fz)
    plt.yticks(fontsize=fz)
    plt.ylabel("Radians", fontsize=fz)
    plt.xlim([0, 50])

    plt.subplot(4, 1, 3)
    # plt.plot(time, state_tpn_dlc[selected_trajectory, 1::4].cpu().detach().numpy(), color="C0")
    # plt.plot(time, state_tpn_dlc[selected_trajectory, 3::4].cpu().detach().numpy(), color="C1")
    # plt.plot(time, state_output_test_batch[selected_trajectory, 1::4].cpu().detach().numpy(), linestyle='dashed', color="C0")
    # plt.plot(time, state_output_test_batch[selected_trajectory, 3::4].cpu().detach().numpy(), linestyle='dashed', color="C1")
    plt.plot(time, state_output_test_batch[selected_trajectory, 1::4].cpu().detach().numpy(), color="C0", linewidth=3)
    plt.plot(time, state_output_test_batch[selected_trajectory, 3::4].cpu().detach().numpy(), color="C1", linewidth=3)
    plt.axhline(0.0, linestyle='dashed', color="black")
    plt.axvspan(0.0, 2.5, color='grey', alpha=0.2)
    plt.axvspan(2.5, 15.0, color=lighten_color('grey', 2.0), alpha=0.2)
    plt.axvspan(15.0, 22.5, color='grey', alpha=0.2)
    plt.axvspan(22.5, 35.0, color=lighten_color('grey', 2.0), alpha=0.2)
    plt.axvspan(35.0, time[-1], color='grey', alpha=0.2)
            # plt.axvline(2.5, linestyle='dashed', color="C2")
            # plt.axvline(15.0, linestyle='dashed', color="C3")
            # plt.axvline(23.5, linestyle='dashed', color="C4")
            # plt.axvline(35.0, linestyle='dashed', color="C5")
            # plt.axvline(49.0, linestyle='dashed', color="C6")
    plt.title("Bob 2 Observation Space", fontsize=fz)
    plt.legend(["$\\theta_{2,t}$", "$\dot{\\theta}_{2,t}$"], fontsize=fz)
    plt.xticks(fontsize=fz)
    plt.yticks(fontsize=fz)
    plt.ylabel("Radians", fontsize=fz)
    plt.xlim([0, 50])

    # Plot latent space prediction
    plt.subplot(4, 1, 2)
    plt.plot(time-1.0, z_tpn_dlc[selected_trajectory, 0::4].cpu().detach().numpy(), linewidth=3)
    plt.plot(time-1.0, z_tpn_dlc[selected_trajectory, 1::4].cpu().detach().numpy(), linewidth=3)
    plt.axvspan(0.0, 11.75, color='grey', alpha=0.2)
    plt.axvspan(11.75, 25.5, color=lighten_color('grey', 2.0), alpha=0.2)
    plt.axvspan(25.5, 37.0, color='grey', alpha=0.2)
    plt.axvspan(37.0, 45.0, color=lighten_color('grey', 2.0), alpha=0.2)
    plt.axvspan(45.0, time[-1], color='grey', alpha=0.2)
            # plt.axvline(12.0, linestyle='dashed', color="C2")
            # plt.axvline(25.5, linestyle='dashed', color="C3")
            # plt.axvline(37.0, linestyle='dashed', color="C4")
            # plt.axvline(45.0, linestyle='dashed', color="C5")
    plt.title("Bob 1 Latent Space", fontsize=fz)
    plt.legend(["$z_{1,1}$", "$z_{1,2}$"], fontsize=fz)
    plt.xticks(fontsize=fz)
    plt.yticks(fontsize=fz)
    plt.ylabel("Affinity", fontsize=fz)
    plt.ylim([-0.2, 0.2])
    plt.xlim([0, 50])

    plt.subplot(4, 1, 4)
    plt.plot(time-1.0, z_tpn_dlc[selected_trajectory, 2::4].cpu().detach().numpy(), linewidth=3)
    plt.plot(time-1.0, z_tpn_dlc[selected_trajectory, 3::4].cpu().detach().numpy(), linewidth=3)
    plt.axvspan(0.0, 2.5, color='grey', alpha=0.2)
    plt.axvspan(2.5, 15.0, color=lighten_color('grey', 2.0), alpha=0.2)
    plt.axvspan(15.0, 22.5, color='grey', alpha=0.2)
    plt.axvspan(22.5, 35.0, color=lighten_color('grey', 2.0), alpha=0.2)
    plt.axvspan(35.0, time[-1], color='grey', alpha=0.2)
            # plt.axvline(2.5, linestyle='dashed', color="C2")
            # plt.axvline(15.0, linestyle='dashed', color="C3")
            # plt.axvline(23.5, linestyle='dashed', color="C4")
            # plt.axvline(35.0, linestyle='dashed', color="C5")
            # plt.axvline(49.0, linestyle='dashed', color="C6")
    plt.title("Bob 2 Latent Space", fontsize=fz)
    plt.legend(["$z_{2,1}$", "$z_{2,2}$"], fontsize=fz)
    plt.xticks(fontsize=fz)
    plt.yticks(fontsize=fz)
    plt.ylabel("Affinity", fontsize=fz)
    plt.xlabel("$t$", fontsize=fz)
    plt.xlim([0, 50])
    
    # Save testing plot PNG
    savepath = 'Model/' + timestr + "/Testing.png"
    savepath_pdf = 'Model/' + timestr + "/Testing.pdf"
    plt.tight_layout(h_pad=2)
    plt.savefig(savepath, bbox_inches='tight')
    plt.savefig(savepath_pdf, bbox_inches='tight')
    # Save testing plot HTML
    savepath = 'Model/' + timestr + "/Testing.html"
    html_str = mpld3.fig_to_html(fig)
    Html_file= open(savepath, "w")
    Html_file.write(html_str)
    Html_file.close()
    
def plot_test_spring(test_loader, state_tpn_dlc, z_tpn_dlc, b_tpn_dlc, dt, selected_trajectory, timestr):
    # Extract ground truth data
    state_input_test_batch, state_output_test_batch = next(iter(test_loader))
    # Initialise plot
    fig = plt.figure(figsize=(18, 10), dpi=100)
    # Initialise time for plotting
    time = np.linspace(1, len(state_output_test_batch[0, 0::20]), len(state_output_test_batch[0, 0::20]))
    # Plot state space prediction
    plt.subplot(1, 2, 1)
    plt.scatter(state_output_test_batch[selected_trajectory, 0].detach().numpy(), state_output_test_batch[selected_trajectory, 5].detach().numpy(), color='C0')
    plt.plot(state_tpn_dlc[selected_trajectory, 0::20].cpu().detach().numpy(), state_tpn_dlc[selected_trajectory, 5::20].cpu().detach().numpy(), color='C0')
    plt.plot(state_output_test_batch[selected_trajectory, 0::20].detach().numpy(), state_output_test_batch[selected_trajectory, 5::20].detach().numpy(), color='C0', linestyle='dashed')
    plt.scatter(state_output_test_batch[selected_trajectory, 1].detach().numpy(), state_output_test_batch[selected_trajectory, 6].detach().numpy(), color='C1')
    plt.plot(state_tpn_dlc[selected_trajectory, 1::20].cpu().detach().numpy(), state_tpn_dlc[selected_trajectory, 6::20].cpu().detach().numpy(), color='C1')
    plt.plot(state_output_test_batch[selected_trajectory, 1::20].detach().numpy(), state_output_test_batch[selected_trajectory, 6::20].detach().numpy(), color='C1', linestyle='dashed')
    plt.scatter(state_output_test_batch[selected_trajectory, 2].detach().numpy(), state_output_test_batch[selected_trajectory, 7].detach().numpy(), color='C2')
    plt.plot(state_tpn_dlc[selected_trajectory, 2::20].cpu().detach().numpy(), state_tpn_dlc[selected_trajectory, 7::20].cpu().detach().numpy(), color='C2')
    plt.plot(state_output_test_batch[selected_trajectory, 2::20].detach().numpy(), state_output_test_batch[selected_trajectory, 7::20].detach().numpy(), color='C2', linestyle='dashed')
    plt.scatter(state_output_test_batch[selected_trajectory, 3].detach().numpy(), state_output_test_batch[selected_trajectory, 8].detach().numpy(), color='C3')
    plt.plot(state_tpn_dlc[selected_trajectory, 3::20].cpu().detach().numpy(), state_tpn_dlc[selected_trajectory, 8::20].cpu().detach().numpy(), color='C3')
    plt.plot(state_output_test_batch[selected_trajectory, 3::20].detach().numpy(), state_output_test_batch[selected_trajectory, 8::20].detach().numpy(), color='C3', linestyle='dashed')
    plt.scatter(state_output_test_batch[selected_trajectory, 4].detach().numpy(), state_output_test_batch[selected_trajectory, 9].detach().numpy(), color='C4')
    plt.plot(state_tpn_dlc[selected_trajectory, 4::20].cpu().detach().numpy(), state_tpn_dlc[selected_trajectory, 9::20].cpu().detach().numpy(), color='C4')
    plt.plot(state_output_test_batch[selected_trajectory, 4::20].detach().numpy(), state_output_test_batch[selected_trajectory, 9::20].detach().numpy(), color='C4', linestyle='dashed')
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    # Plot latent space prediction
    plt.subplot(2, 4, 3)
    plt.plot(time, z_tpn_dlc[selected_trajectory, 0::20].cpu().detach().numpy(), color='C0')
    plt.plot(time, z_tpn_dlc[selected_trajectory, 4::20].cpu().detach().numpy(), color='C1')
    plt.plot(time, z_tpn_dlc[selected_trajectory, 8::20].cpu().detach().numpy(), color='C2')
    plt.plot(time, z_tpn_dlc[selected_trajectory, 12::20].cpu().detach().numpy(), color='C3')
    plt.plot(time, z_tpn_dlc[selected_trajectory, 16::20].cpu().detach().numpy(), color='C4')
    plt.title("$z_{i1}$")
    plt.xlabel("$t$")
    plt.subplot(2, 4, 7)
    plt.plot(time, z_tpn_dlc[selected_trajectory, 1::20].cpu().detach().numpy(), color='C0')
    plt.plot(time, z_tpn_dlc[selected_trajectory, 5::20].cpu().detach().numpy(), color='C1')
    plt.plot(time, z_tpn_dlc[selected_trajectory, 9::20].cpu().detach().numpy(), color='C2')
    plt.plot(time, z_tpn_dlc[selected_trajectory, 13::20].cpu().detach().numpy(), color='C3')
    plt.plot(time, z_tpn_dlc[selected_trajectory, 17::20].cpu().detach().numpy(), color='C4')
    plt.title("$z_{i2}$")
    plt.xlabel("$t$")
    plt.subplot(2, 4, 4)
    plt.plot(time, z_tpn_dlc[selected_trajectory, 2::20].cpu().detach().numpy(), color='C0')
    plt.plot(time, z_tpn_dlc[selected_trajectory, 6::20].cpu().detach().numpy(), color='C1')
    plt.plot(time, z_tpn_dlc[selected_trajectory, 10::20].cpu().detach().numpy(), color='C2')
    plt.plot(time, z_tpn_dlc[selected_trajectory, 14::20].cpu().detach().numpy(), color='C3')
    plt.plot(time, z_tpn_dlc[selected_trajectory, 18::20].cpu().detach().numpy(), color='C4')
    plt.title("$z_{i3}$")
    plt.xlabel("$t$")
    plt.subplot(2, 4, 8)
    plt.plot(time, z_tpn_dlc[selected_trajectory, 3::20].cpu().detach().numpy(), color='C0')
    plt.plot(time, z_tpn_dlc[selected_trajectory, 7::20].cpu().detach().numpy(), color='C1')
    plt.plot(time, z_tpn_dlc[selected_trajectory, 11::20].cpu().detach().numpy(), color='C2')
    plt.plot(time, z_tpn_dlc[selected_trajectory, 15::20].cpu().detach().numpy(), color='C3')
    plt.plot(time, z_tpn_dlc[selected_trajectory, 19::20].cpu().detach().numpy(), color='C4')
    plt.title("$z_{i4}$")
    plt.xlabel("$t$")
    # Save testing plot PNG
    savepath = 'Model/' + timestr + "/Testing.png"
    plt.savefig(savepath)
    # Save testing plot HTML
    savepath = 'Model/' + timestr + "/Testing.html"
    html_str = mpld3.fig_to_html(fig)
    Html_file= open(savepath, "w")
    Html_file.write(html_str)
    Html_file.close()

def plot_test_kuramoto(test_loader, state_tpn_dlc, z_tpn_dlc, b_tpn_dlc, dt, selected_trajectory, timestr):
    # Extract ground truth data
    state_input_test_batch, state_output_test_batch = next(iter(test_loader))
    # Initialise plot
    fig = plt.figure(figsize=(18, 10), dpi=100)
    # Initialise time for plotting
    time = np.linspace(1, len(state_output_test_batch[0, 0::5]), len(state_output_test_batch[0, 0::5]))
    # Plot state space prediction
    plt.scatter(0, torch.sin(state_input_test_batch[selected_trajectory, 0]).detach().numpy(), color='C0')
    plt.plot(time, torch.sin(state_tpn_dlc[selected_trajectory, 0::5]).cpu().detach().numpy(), color='C0')
    plt.plot(time, torch.sin(state_output_test_batch[selected_trajectory, 0::5]).detach().numpy(), color='C0', linestyle='dashed')
    plt.scatter(0, torch.sin(state_input_test_batch[selected_trajectory, 1]).detach().numpy(), color='C1')
    plt.plot(time, torch.sin(state_tpn_dlc[selected_trajectory, 1::5]).cpu().detach().numpy(), color='C1')
    plt.plot(time, torch.sin(state_output_test_batch[selected_trajectory, 1::5]).detach().numpy(), color='C1', linestyle='dashed')
    plt.scatter(0, torch.sin(state_input_test_batch[selected_trajectory, 2]).detach().numpy(), color='C2')
    plt.plot(time, torch.sin(state_tpn_dlc[selected_trajectory, 2::5]).cpu().detach().numpy(), color='C2')
    plt.plot(time, torch.sin(state_output_test_batch[selected_trajectory, 2::5]).detach().numpy(), color='C2', linestyle='dashed')
    plt.scatter(0, torch.sin(state_input_test_batch[selected_trajectory, 3]).detach().numpy(), color='C3')
    plt.plot(time, torch.sin(state_tpn_dlc[selected_trajectory, 3::5]).cpu().detach().numpy(), color='C3')
    plt.plot(time, torch.sin(state_output_test_batch[selected_trajectory, 3::5]).detach().numpy(), color='C3', linestyle='dashed')
    plt.scatter(0, torch.sin(state_input_test_batch[selected_trajectory, 4]).detach().numpy(), color='C4')
    plt.plot(time, torch.sin(state_tpn_dlc[selected_trajectory, 4::5]).cpu().detach().numpy(), color='C4')
    plt.plot(time, torch.sin(state_output_test_batch[selected_trajectory, 4::5]).detach().numpy(), color='C4', linestyle='dashed')
    plt.xlabel("$t$")
    plt.ylabel("$y$")
    # Save testing plot PNG
    savepath = 'Model/' + timestr + "/Testing.png"
    plt.savefig(savepath)
    # Save testing plot HTML
    savepath = 'Model/' + timestr + "/Testing.html"
    html_str = mpld3.fig_to_html(fig)
    Html_file= open(savepath, "w")
    Html_file.write(html_str)
    Html_file.close()

def plot_test(model_name, test_loader, state_tpn_dlc, z_tpn_dlc, b_tpn_dlc, dt, selected_trajectory, timestr):
    if model_name == "IP":
        plot_test_pendulum(test_loader, state_tpn_dlc, z_tpn_dlc, b_tpn_dlc, dt, selected_trajectory, timestr)
    if model_name == "DP2":
        plot_test_doublependulum(test_loader, state_tpn_dlc, z_tpn_dlc, b_tpn_dlc, dt, selected_trajectory, timestr)
    if model_name == "MS5":
        plot_test_spring(test_loader, state_tpn_dlc, z_tpn_dlc, b_tpn_dlc, dt, selected_trajectory, timestr)
    if model_name == "KM5":
        plot_test_kuramoto(test_loader, state_tpn_dlc, z_tpn_dlc, b_tpn_dlc, dt, selected_trajectory, timestr)

def plot_test_grid(test_loader, state_tpn_dlc, z_tpn_dlc, b_tpn_dlc, dt, selected_trajectory, timestr):
    # Extract ground truth data
    state_input_test_batch, state_output_test_batch = next(iter(test_loader))
    # Initialise plot
    fig = plt.figure(figsize=(18, 10), dpi=100)
    # Initialise time for plotting
    time = np.linspace(1, len(state_output_test_batch[0, 0::20]), len(state_output_test_batch[0, 0::20]))
    # Loop over six plots
    for i in range(6):
        plt.subplot(2, 3, i+1)
        plt.plot(state_tpn_dlc[selected_trajectory+i*10, 0::20].cpu().detach().numpy(), state_tpn_dlc[selected_trajectory+i*10, 5::20].cpu().detach().numpy(), color='C0')
        plt.plot(state_tpn_dlc[selected_trajectory+i*10, 1::20].cpu().detach().numpy(), state_tpn_dlc[selected_trajectory+i*10, 6::20].cpu().detach().numpy(), color='C1')
        plt.plot(state_tpn_dlc[selected_trajectory+i*10, 2::20].cpu().detach().numpy(), state_tpn_dlc[selected_trajectory+i*10, 7::20].cpu().detach().numpy(), color='C2')
        plt.plot(state_tpn_dlc[selected_trajectory+i*10, 3::20].cpu().detach().numpy(), state_tpn_dlc[selected_trajectory+i*10, 8::20].cpu().detach().numpy(), color='C3')
        plt.plot(state_tpn_dlc[selected_trajectory+i*10, 4::20].cpu().detach().numpy(), state_tpn_dlc[selected_trajectory+i*10, 9::20].cpu().detach().numpy(), color='C4')
        plt.scatter(state_input_test_batch[selected_trajectory+i*10, 0].cpu().detach().numpy(), state_input_test_batch[selected_trajectory+i*10, 5].cpu().detach().numpy(), color='C0')
        plt.plot(state_output_test_batch[selected_trajectory+i*10, 0::20].cpu().detach().numpy(), state_output_test_batch[selected_trajectory+i*10, 5::20].cpu().detach().numpy(), color='C0', linestyle='dashed')
        plt.scatter(state_input_test_batch[selected_trajectory+i*10, 1].cpu().detach().numpy(), state_input_test_batch[selected_trajectory+i*10, 6].cpu().detach().numpy(), color='C1')
        plt.plot(state_output_test_batch[selected_trajectory+i*10, 1::20].cpu().detach().numpy(), state_output_test_batch[selected_trajectory+i*10, 6::20].cpu().detach().numpy(), color='C1', linestyle='dashed')
        plt.scatter(state_input_test_batch[selected_trajectory+i*10, 2].cpu().detach().numpy(), state_input_test_batch[selected_trajectory+i*10, 7].cpu().detach().numpy(), color='C2')
        plt.plot(state_output_test_batch[selected_trajectory+i*10, 2::20].cpu().detach().numpy(), state_output_test_batch[selected_trajectory+i*10, 7::20].cpu().detach().numpy(), color='C2', linestyle='dashed')
        plt.scatter(state_input_test_batch[selected_trajectory+i*10, 3].cpu().detach().numpy(), state_input_test_batch[selected_trajectory+i*10, 8].cpu().detach().numpy(), color='C3')
        plt.plot(state_output_test_batch[selected_trajectory+i*10, 3::20].cpu().detach().numpy(), state_output_test_batch[selected_trajectory+i*10, 8::20].cpu().detach().numpy(), color='C3', linestyle='dashed')
        plt.scatter(state_input_test_batch[selected_trajectory+i*10, 4].cpu().detach().numpy(), state_input_test_batch[selected_trajectory+i*10, 9].cpu().detach().numpy(), color='C4')
        plt.plot(state_output_test_batch[selected_trajectory+i*10, 4::20].cpu().detach().numpy(), state_output_test_batch[selected_trajectory+i*10, 9::20].cpu().detach().numpy(), color='C4', linestyle='dashed')
    # Plotting Axis
    plt.subplot(2, 3, 1); plt.ylabel("$y$")
    plt.subplot(2, 3, 4); plt.ylabel("$x$"); plt.ylabel("$y$")
    plt.subplot(2, 3, 5); plt.ylabel("$x$")
    plt.subplot(2, 3, 6); plt.ylabel("$x$")
    # Save testing plot PNG
    savepath = 'Model/' + timestr + "/Testing_Grid.png"
    plt.savefig(savepath)
    # Save testing plot HTML
    savepath = 'Model/' + timestr + "/Testing_Grid.html"
    html_str = mpld3.fig_to_html(fig)
    Html_file= open(savepath, "w")
    Html_file.write(html_str)
    Html_file.close()