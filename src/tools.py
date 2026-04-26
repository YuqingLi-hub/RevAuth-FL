import copy
import torch
import types
import math
import numpy as np
from torch.nn.utils import vector_to_parameters
# --------------------------------------------------------------------- #
# Gradient access

def grad_of(tensor):
    """ Get the gradient of a given tensor, make it zero if missing.
    Args:
    tensor Given instance of/deriving from Tensor
    Returns:
    Gradient for the given tensor
    """
    # Get the current gradient
    grad = tensor.grad
    if grad is not None:
        return grad
    # Make and set a zero-gradient
    grad = torch.zeros_like(tensor)
    tensor.grad = grad
    return grad

def grads_of(tensors):
    """ Iterate of the gradients of the given tensors, make zero gradients if missing.
    Args:
    tensors Generator of/iterable on instances of/deriving from Tensor
    Returns:
    Generator of the gradients of the given tensors, in emitted order
    """
    return (grad_of(tensor) for tensor in tensors)

# ---------------------------------------------------------------------------- #
# "Flatten" and "relink" operations

def relink(tensors, common):
    """ "Relink" the tensors of class (deriving from) Tensor by making them point to another contiguous segment of memory.
    Args:
    tensors Generator of/iterable on instances of/deriving from Tensor, all with the same dtype
    common  Flat tensor of sufficient size to use as underlying storage, with the same dtype as the given tensors
    Returns:
    Given common tensor
    """
    # Convert to tuple if generator
    if isinstance(tensors, types.GeneratorType):
        tensors = tuple(tensors)
    # Relink each given tensor to its segment on the common one
    pos = 0
    for tensor in tensors:
        npos = pos + tensor.numel()
        tensor.data = common[pos:npos].view(*tensor.shape)
        pos = npos
    # Finalize and return
    common.linked_tensors = tensors
    return common

def flatten(tensors):
    """ "Flatten" the tensors of class (deriving from) Tensor so that they all use the same contiguous segment of memory.
    Args:
    tensors Generator of/iterable on instances of/deriving from Tensor, all with the same dtype
    Returns:
    Flat tensor (with the same dtype as the given tensors) that contains the memory used by all the given Tensor (or derived instances), in emitted order
    """
    # Convert to tuple if generator
    if isinstance(tensors, types.GeneratorType):
        tensors = tuple(tensors)
    # Common tensor instantiation and reuse
    common = torch.cat(tuple(tensor.view(-1) for tensor in tensors))
    # Return common tensor
    return relink(tensors, common)

# ---------------------------------------------------------------------------- #

def get_gradient(model):
    """ Get (optionally make each parameter's gradient) a reference to the flat gradient.
    Returns:
      Flat gradient (by reference: future calls to 'set_gradient' will modify it)
    """
    # Flatten (make if necessary)
    gradient = flatten(grads_of(model.parameters()))
    return gradient

def set_gradient(model, gradient):
    """ Overwrite the gradient with the given one.
    Args:
      gradient Given flat gradient
    """
    # Assignment
    grad_old = get_gradient(model)
    grad_old.copy_(gradient)

def get_gradient_values(model):
    """ Get (optionally make each parameter's gradient) a reference to the flat gradient.
    Returns:
      Flat gradient (by reference: future calls to 'set_gradient' will modify it)
    """

    # gradient = torch.cat([torch.reshape(param.grad, (-1,)) for param in model.parameters()]).clone().detach()
    # return gradient
    grads = []
    for param in model.parameters():
        if param.grad is None:
            grads.append(torch.zeros_like(param).flatten())  # fill missing grads with zeros
        else:
            grads.append(param.grad.flatten())
    return torch.cat(grads).clone().detach()
    

def set_gradient_values(model, gradient):
    """ Overwrite the gradient with the given one.
    Args:
      gradient Given flat gradient
    """
    cur_pos = 0
    for param in model.parameters():
        param.grad = torch.reshape(torch.narrow(gradient, 0, cur_pos, param.nelement()), param.size()).clone().detach()
        cur_pos = cur_pos + param.nelement()

def get_parameter_values(model):
    """ Get (optionally make each parameter's gradient) a reference to the flat gradient.
    Returns:
      Flat gradient (by reference: future calls to 'set_gradient' will modify it)
    """

    parameter = torch.cat([torch.reshape(param.data, (-1,)) for param in model.parameters()]).clone().detach()
    return parameter
# ---------------------------------------------------------------------------- #
# Simple generator on the pairs (x, y) of an indexable such that index x < index y

def pairwise(data):
    """ Simple generator of the pairs (x, y) in a tuple such that index x < index y.
    Args:
    data Indexable (including ability to query length) containing the elements
    Returns:
    Generator over the pairs of the elements of 'data'
    """
    n = len(data)
    for i in range(n - 1):
        for j in range(i + 1, n):
            yield (data[i], data[j])

def elementwise_distance(data, point):
    '''
    :param data: tensor of gradients
    :param point: tensor of gradient
    :return: distance tensor
    '''
    return torch.norm(data-point,dim=1)

def pairwise_distance(data):
    device = data[0][0].device
    n = data.shape[0]
    distances = []
    for i in range(n):
        dist = elementwise_distance(data, data[i])
        distances.append(dist)
    return torch.stack(distances)


def pairwise_distance_faster(gradients):
    device = gradients[0][0].device
    n = gradients.shape[0]
    distances = torch.zeros((n,n),device=device)
    for gid_x, gid_y in pairwise(tuple(range(n))):
        dist = gradients[gid_x].sub(gradients[gid_y]).norm().item()
        if not math.isfinite(dist):
            dist = math.inf
        distances[gid_x][gid_y] = dist
        distances[gid_y][gid_x] = dist
    return distances


def max_pairwise_distance(data):
    '''
    :param data: tensor of gradients
    :return: max distance value
    '''
    n = data.shape[0]
    distances = [0] * (n * (n - 1) // 2)
    if n <= 1:
        distances = [0]
    for i, (x, y) in enumerate(pairwise(tuple(range(n)))):
        dist = data[x].sub(data[y]).norm().item()
        if not math.isfinite(dist):
            dist = math.inf
        distances[i] = dist
    return max(distances)


def sum_distance(data, point):
    '''
    :param data: tensor
    :param point:
    :return:
    '''
    return elementwise_distance(data, point).sum().item()


def max_distance(data, point):
    '''
    :param data: tensor
    :param point:
    :return:
    '''
    return elementwise_distance(data, point).max().item()


def max_sum_distance(data):
    '''
    :param data: tensor
    :return:
    '''
    n = data.shape[0]
    dist_sum = [0]*n
    for i in range(n):
        dist_sum[i] = elementwise_distance(data, data[i]).sum().item()
    return max(dist_sum)


def pairwise_similarity(data):
    device = data[0][0].device
    n = data.shape[0]
    similarity = []
    for i in range(n):
        sim = torch.cosine_similarity(data, data[i], dim=-1)
        similarity.append(sim)
    return torch.stack(similarity)


def pairwise_similarity_faster(gradients):
    device = gradients[0][0].device
    n = gradients.shape[0]
    similarity = torch.zeros((n,n),device=device)
    for gid_x, gid_y in pairwise(tuple(range(n))):
        sim = torch.cosine_similarity(gradients[gid_x], gradients[gid_y], dim=-1)
        similarity[gid_x][gid_y] = sim
        similarity[gid_y][gid_x] = sim
    return similarity


def pairwise_sign_similarity(data):
    device = data[0][0].device
    n = data.shape[0]
    d = data.shape[1]
    similarity = []
    for i in range(n):
        sim = data.eq(data[i]).sum(dim=-1).float() - data.ne(data[i]).sum(dim=-1).float()
        similarity.append(sim/d)
    return torch.stack(similarity)


def pairwise_sign_similarity_plus(data):
    device = data[0][0].device
    n = data.shape[0]
    d = data.shape[1]
    similarity = []
    for i in range(n):
        pos_idx = torch.nonzero(data[i].eq(1.0)).squeeze()
        zero_idx = torch.nonzero(data[i].eq(0.0)).squeeze()
        neg_idx = torch.nonzero(data[i].eq(-1.0)).squeeze()
        sim_p = data[:,pos_idx].eq(data[i][pos_idx]).sum(dim=-1).float()
        sim_0 = data[:,zero_idx].eq(data[i][zero_idx]).sum(dim=-1).float()
        sim_n = data[:,neg_idx].eq(data[i][neg_idx]).sum(dim=-1).float()
        similarity.append((sim_p+sim_0+sim_n)/d)
    return torch.stack(similarity)

# -------------------------------------------------------------------------- #


def embedding_watermark_on_position(masks,whole_grads,Watermark,message,alpha,k,model=None):
    # device = args.device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # alpha = args.alpha
    # k = args.k
    # delta = args.delta
    # print('Alpha used in embedding: ', alpha, delta, k)

    # Extract the section to watermark
    grad_unwater = whole_grads[masks[0]:masks[1]]
    grad_unwater = grad_unwater.detach().cpu()
    w_ = Watermark.embed(grad_unwater, m=message, alpha=alpha, k=k).to(device)

    # Update the flat tensor
    whole_grads[masks[0]:masks[1]].copy_(w_)

    # If model is provided, update the actual model parameters in-place
    if model is not None:
        with torch.no_grad():
            vector_to_parameters(whole_grads,model.parameters())
            # start = 0
            # for p in model.parameters():
            #     numel = p.numel()
            #     p.data.copy_(whole_grads[start:start+numel].view_as(p))
            #     start += numel
    return whole_grads

def detect_recover_on_position(masks,whole_grads,Watermark,alpha,k,model=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # alpha = args.alpha
    # k = args.k
    # delta = args.delta
    # print('Alpha used in detecting: ',alpha,delta,k)
    grad_water = copy.deepcopy(whole_grads[masks[0]:masks[1]])
    grad_water = grad_water.detach().cpu()
    r_w,mm = Watermark.detect(grad_water,alpha=alpha,k=k)

    reconstructed_grad = r_w.to(device)
    whole_grads[masks[0]:masks[1]].copy_(reconstructed_grad)
    if model is not None:
        with torch.no_grad():
            vector_to_parameters(whole_grads,model.parameters())
            # start = 0
            # for p in model.parameters():
            #     numel = p.numel()
            #     p.data.copy_(whole_grads[start:start+numel].view_as(p))
            #     start += numel
    return whole_grads, mm.to(device)
def update_gradient(whole_grads, model):
    with torch.no_grad():
        start = 0
        for p in model.parameters():
            numel = p.numel()
            g = whole_grads[start:start+numel].view_as(p)
            if p.grad is not None:
                p.grad.copy_(g)  
            start += numel

def vector_to_model(vec, model):
    # Pointer for slicing the vector for each parameter
    state_dict = model.state_dict()
    pointer = 0
    for name in state_dict:
        # The length of the parameter
        num_param = state_dict[name].numel()
        # Slice the vector, reshape it, and replace the old data of the parameter
        state_dict[name].data = vec[pointer:pointer + num_param].view_as(state_dict[name]).data
        # Increment the pointer
        pointer += num_param
    model.load_state_dict(state_dict)
    return state_dict

def vector_to_model_wo_load(vec, model):
    # Pointer for slicing the vector for each parameter
    state_dict = model.state_dict()
    pointer = 0
    for name in state_dict:
        # The length of the parameter
        num_param = state_dict[name].numel()
        # Slice the vector, reshape it, and replace the old data of the parameter
        state_dict[name].data = vec[pointer:pointer + num_param].view_as(state_dict[name]).data
        # Increment the pointer
        pointer += num_param

    return state_dict

if __name__ == "__main__":
    from watermarks.modi_qim import QIM
    from options import args_parser
    delta = 0.1
    Watermark = QIM(delta=delta)
    message = Watermark.random_msg(l=5)
    alpha = 0.1
    k = 4
    t_ = torch.randn(10)
    print('Original Gradient: ',t_[:10])
    # w_ = Watermark.embed(t_,m=message,alpha=alpha,k=k)
    args = args_parser()
    w_ = embedding_watermark_on_position([0,5],t_,Watermark,message,args)
    print('Watermarked Gradient: ',w_[:10])
    # r_w,mm = Watermark.detect(w_,alpha=alpha,k=k)
    r_w,mm = detect_recover_on_position([0,5],t_,Watermark,args)
    print('Recovered Gradient: ',r_w[:10])
    print('Recovery success: ', np.allclose(t_,r_w))
    print('Message recovery success: ', message==mm)