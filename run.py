import torch, numpy as np, matplotlib.pyplot as plt
import tqdm, random, argparse, time, sys

from model_utils import train_epoch, test_epoch, validate_epoch, initialise_DLC_model, save_DLC_model, load_DLC_model
from plot_utils import plot_loss_figure, print_lowest_loss, print_NOD_parameters, plot_test, write_sh, append_loss
from data_utils import load_dataset

torch.autograd.set_detect_anomaly(True)
CUDA_percision = "highest"
torch.set_float32_matmul_precision(CUDA_percision)
if CUDA_percision == "highest":
    print("WARNING: CUDA MatMul Percision Set to highest which may lead to lower training performance!")
elif CUDA_percision == "high":
    print("WARNING: CUDA MatMul Percision Set to high which may lead to lower model accuracy!")
elif CUDA_percision == "medium":
    print("WARNING: CUDA MatMul Percision Set to medium which may lead to lower model accuracy!")

if torch.cuda.is_available():
    print("Training on " + torch.cuda.get_device_name(0))
    device = torch.device('cuda')
else:
    print("WARNING: Training on CPU")
    device = torch.device('cpu')

def train(epoch, data_offset, dt, nAgent, nOpinion, nDim, dOrder, batch_size, encoder, NOD_func, decoder, optimiser, scheduler, train_loader, valid_loader, timestr, model_name, inverse):
    # Intialise loss arrays
    loss_train_log, loss_pred_train_log, loss_recon_train_log, loss_latent_train_log, loss_pred_train_lowest = [], [], [], [], 10000
    loss_valid_log, loss_pred_valid_log, loss_recon_valid_log, loss_latent_valid_log, loss_pred_valid_lowest = [], [], [], [], 10000
    # Train DLC model
    print("Training DLC Model " + model_name +  " for " + str(epoch) + " Total Epochs.")
    for i in tqdm.tqdm(range(epoch)):
        # Train model
        loss_train, loss_pred_train, loss_recon_train, loss_latent_train = train_epoch(encoder, NOD_func, decoder, optimiser, scheduler, device, train_loader, data_offset, nAgent, nOpinion, nDim, dOrder, batch_size, dt, inverse)
        # Print options and log training loss
        append_loss(loss_train, loss_pred_train, loss_recon_train, loss_latent_train, loss_train_log, loss_pred_train_log, loss_recon_train_log, loss_latent_train_log, train_loader, "Train")
        # Validate model
        loss_valid, loss_pred_valid, loss_recon_valid, loss_latent_valid = validate_epoch(encoder, NOD_func, decoder, device, valid_loader, data_offset, nAgent, nOpinion, nDim, dOrder, batch_size, dt, inverse)
        # # Print options and log validation loss
        append_loss(loss_valid, loss_pred_valid, loss_recon_valid, loss_latent_valid, loss_valid_log, loss_pred_valid_log, loss_recon_valid_log, loss_latent_valid_log, valid_loader, "Validation")
        # Save better performing model
        loss_pred_valid_lowest, loss_pred_train_lowest = save_DLC_model(loss_pred_valid, loss_pred_valid_lowest, valid_loader, loss_pred_train, loss_pred_train_lowest, train_loader, encoder, NOD_func, decoder, timestr)
    # Plot training loss curves
    plot_loss_figure(epoch, loss_train_log, loss_pred_train_log, loss_recon_train_log, loss_latent_train_log, timestr, name="Training")
    # Plot validation loss curves
    plot_loss_figure(epoch, loss_valid_log, loss_pred_valid_log, loss_recon_valid_log, loss_latent_valid_log, timestr, name="Validation")
    # Save lowest training and validation loss values
    print_lowest_loss(loss_pred_valid_lowest, loss_pred_train_lowest, timestr)

def test(data_offset, dt, nAgent, nOpinion, nDim, dOrder, encoder, NOD_func, decoder, test_loader, timestr, model_name, inverse):
    # Test DLC model
    loss_test, loss_pred_test, loss_recon_test, loss_latent_test, state_tpn_dlc, z_tpn_dlc, b_tpn_dlc = test_epoch(encoder, NOD_func, decoder, device, test_loader, data_offset, nAgent, nOpinion, nDim, dOrder, dt, inverse)
    # Plot testing results
    plot_test(model_name, test_loader, state_tpn_dlc, z_tpn_dlc, b_tpn_dlc, dt, selected_trajectory=15, timestr=timestr)
    # Print learned NOD parameter
    print_NOD_parameters(NOD_func, nOpinion, nAgent, device, timestr)

def main(args):
    # Set training parameters
    seed = args["seed"]
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    epoch = args["epoch"]
    arch= args["arch"]
    learning_rate = args["learning_rate"]
    batch_size = args["batch_size"]
    scheduler_step = args["scheduler_step"]
    scheduler_gamma = args["scheduler_gamma"]
    num_worker = args["num_worker"]
    activation = args["activation"]
    train_set = args["train_set"]
    test_folder = args["test_folder"]
    train_option = args["train"]
    test_option = args["test"]
    inverse = args["inverse"]
    dataset_name = args["dataset"]
    # Set model saving directory
    timestr = time.strftime("%Y%m%d-%H%M%S")
    # Load dataset
    train_loader, valid_loader, test_loader, nAgent, nDim, dt, data_offset = load_dataset(dataset_name, batch_size, train_set, plot_option=True, num_worker=num_worker)
    # Set model parameter
    nOpinion = args["nOpinion"]
    dOrder = args["dOrder"]
    in_dim = dOrder*nAgent*nDim
    hid_dim = args["hid_dim"]
    # Train DLC model
    if train_option: 
        # Save run parameters
        write_sh(args=args, timestr=timestr)
        # Initilise DLC model
        encoder, NOD_func, decoder, optimiser, scheduler = initialise_DLC_model(in_dim, hid_dim, nAgent, nOpinion, nDim, dOrder, device, learning_rate, scheduler_step, scheduler_gamma, arch, activation)
        # Train DLC model
        train(epoch, data_offset, dt, nAgent, nOpinion, nDim, dOrder, batch_size, encoder, NOD_func, decoder, optimiser, scheduler, train_loader, valid_loader, timestr, dataset_name, inverse)
    # Test DLC model
    if test_option:
        # Set loading pathway
        if train_option:
            timestr = timestr
        else:
            timestr = test_folder
        # Load trained DLC model
        encoder, NOD_func, decoder = load_DLC_model(in_dim, hid_dim, nAgent, nOpinion, nDim, dOrder, device, timestr, arch, activation)
        # Test DLC model
        test(data_offset, dt, nAgent, nOpinion, nDim, dOrder, encoder, NOD_func, decoder, test_loader, timestr, dataset_name, inverse)

if __name__ == "__main__":
    # Set global print options
    np.set_printoptions(precision=6, threshold=10_000)
    torch.set_printoptions(precision=6, threshold=10_000, sci_mode=False)
    # Parse run options
    options = argparse.ArgumentParser()
    # Learning parameter
    options.add_argument("--seed", type=int, default=72)
    options.add_argument("--arch", type=str, default='MPNN')
    options.add_argument("--epoch", type=int, default=100)
    options.add_argument("--learning_rate", type=float, default=1E-3)
    options.add_argument("--batch_size", type=int, default=16)
    options.add_argument("--scheduler_step", type=int, default=20)
    options.add_argument("--scheduler_gamma", type=float, default=0.1)
    options.add_argument("--activation", type=str, default='tanh')
    options.add_argument("--num_worker", type=int, default=24)
    # Data generation parameter
    options.add_argument("--dataset", type=str, default='IP')
    options.add_argument("--train_set", type=int, default=50000)
    options.add_argument("--test_folder", type=str, default=str())
    # Run options
    options.add_argument("--train", type=int, default=1)
    options.add_argument("--test", type=int, default=1)
    options.add_argument("--hid_dim", type=int, default=128)
    options.add_argument("--nOpinion", type=int, default=2)
    options.add_argument("--dOrder", type=int, default=2)
    options.add_argument("--inverse", type=int, default=0)
    # Parse arguements
    args = vars(options.parse_args())
    # Train neural net
    main(args)
    # Show plots
    plt.show()