from mpi4py import MPI
import numpy as np
import torch

# sync_networks across the different cores
def sync_networks(network):
    """
    netowrk is the network you want to sync

    """
    comm = MPI.COMM_WORLD
    flat_params, params_shape = _get_flat_params(network)
    comm.Bcast(flat_params, root=0)
    # set the flat params back to the network
    _set_flat_params(network, params_shape, flat_params)

# get the flat params from the network
def _get_flat_params(network):
    param_shape = {}
    flat_params = None
    for key_name, value in network.named_parameters():
        param_shape[key_name] = value.detach().numpy().shape
        if flat_params is None:
            flat_params = value.detach().numpy().flatten()
        else:
            flat_params = np.append(flat_params, value.detach().numpy().flatten())
    return flat_params, param_shape

# set the params from the network
def _set_flat_params(network, params_shape, params):
    pointer = 0
    for key_name, values in network.named_parameters():
        # get the length of the parameters
        len_param = np.prod(params_shape[key_name])
        copy_params = params[pointer:pointer + len_param].reshape(params_shape[key_name])
        copy_params = torch.tensor(copy_params)
        # copy the params
        values.data.copy_(copy_params.data)
        # update the pointer
        pointer += len_param

# sync the networks
def sync_grads(network):
    flat_grads, grads_shape = _get_flat_grads(network)
    comm = MPI.COMM_WORLD
    global_grads = np.zeros_like(flat_grads)
    comm.Allreduce(flat_grads, global_grads, op=MPI.SUM)
    _set_flat_grads(network, grads_shape, global_grads)

def _set_flat_grads(network, grads_shape, flat_grads):
    pointer = 0
    for key_name, value in network.named_parameters():
        len_grads = np.prod(grads_shape[key_name])
        copy_grads = flat_grads[pointer:pointer + len_grads].reshape(grads_shape[key_name])
        copy_grads = torch.tensor(copy_grads)
        # copy the grads
        value.grad.data.copy_(copy_grads.data)
        pointer += len_grads

def _get_flat_grads(network):
    grads_shape = {}
    flat_grads = None
    for key_name, value in network.named_parameters():
        grads_shape[key_name] = value.grad.data.cpu().numpy().shape
        if flat_grads is None:
            flat_grads = value.grad.data.cpu().numpy().flatten()
        else:
            flat_grads = np.append(flat_grads, value.grad.data.cpu().numpy().flatten())
    return flat_grads, grads_shape
