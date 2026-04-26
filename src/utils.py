import copy
import logging
import os

import torch
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils import vector_to_parameters
from torch.nn.utils import vector_to_parameters
from torchvision import datasets, transforms
from math import floor
from collections import defaultdict
import random
import math
import time
from shutil import copyfile
import pprint


class DatasetSplit(Dataset):
    """ An abstract Dataset class wrapped around Pytorch Dataset class """

    def __init__(self, dataset, idxs, runtime_poison=False, args=None, client_id=-1, modify_label=True):
        self.dataset = dataset
        self.idxs = idxs
        self.targets = torch.Tensor([self.dataset.targets[idx] for idx in idxs])
        self.runtime_poison = runtime_poison
        self.args = args
        self.client_id = client_id
        self.modify_label = modify_label
        if client_id == -1:
            poison_frac = 1
        elif client_id < self.args.num_corrupt:
            poison_frac = self.args.poison_frac
        else:
            poison_frac = 0
        self.poison_sample = {}
        self.poison_idxs = []
        if runtime_poison and poison_frac > 0:
            self.poison_idxs = random.sample(self.idxs, floor(poison_frac * len(self.idxs)))
            for idx in self.poison_idxs:
                self.poison_sample[idx] = add_pattern_bd(copy.deepcopy(self.dataset[idx][0]), None, args.data,
                                                         pattern_type=args.pattern_type, agent_idx=client_id,
                                                         attack=args.attack)
                # plt.imshow(self.poison_sample[idx].permute(1, 2, 0))
                # plt.show()

    def classes(self):
        return torch.unique(self.targets)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        # print(target.type())
        if self.idxs[item] in self.poison_idxs:
            inp = self.poison_sample[self.idxs[item]]
            if self.modify_label:
                target = self.args.target_class
            else:
                target = self.dataset[self.idxs[item]][1]
        else:
            inp, target = self.dataset[self.idxs[item]]

        return inp, target


class DatasetSplit_new(Dataset):
    """ An abstract Dataset class wrapped around Pytorch Dataset class """

    def __init__(self, dataset, backdoor_dataset, benign_idx, backdoor_idx, idxs, runtime_poison=False, args=None, client_id=-1, modify_label=True):
        self.dataset = dataset
        self.backdoor_dataset = backdoor_dataset
        self.backdoor_dataset.targets = torch.ones_like(self.backdoor_dataset.targets)
        self.benign_idx = benign_idx
        self.backdoor_idx = backdoor_idx
        self.idxs = idxs
        self.targets = torch.Tensor(self.dataset.targets.float())
        self.runtime_poison = runtime_poison
        self.args = args
        self.client_id = client_id
        self.modify_label = modify_label
        if client_id == -1:
            poison_frac = 1
        elif client_id < self.args.num_corrupt:
            poison_frac = self.args.poison_frac
        else:
            poison_frac = 0
        self.poison_sample = {}
        self.poison_idxs = []
                # plt.imshow(self.poison_sample[idx].permute(1, 2, 0))
                # plt.show()

    def classes(self):
        return torch.unique(self.targets)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        # print(target.type())
        # print(item)
        # print(self.benign_idx)
        # print(self.backdoor_idx)
        if item not in self.benign_idx:
            inp, target = self.backdoor_dataset[self.idxs[item]]
            target = torch.ones_like(target)
            # print('yes')
        else:
            inp, target = self.dataset[self.idxs[item]]
        # inp, target = self.dataset[self.idxs[item]]

        return inp, target


def distribute_data_dirichlet(dataset, args):
    # sort labels
    labels_sorted = dataset.targets.sort()
    # create a list of pairs (index, label), i.e., at index we have an instance of  label
    class_by_labels = list(zip(labels_sorted.values.tolist(), labels_sorted.indices.tolist()))
    labels_dict = defaultdict(list)

    for k, v in class_by_labels:
        labels_dict[k].append(v)
    # convert list to a dictionary, e.g., at labels_dict[0], we have indexes for class 0
    N = len(labels_sorted[1])
    K = len(labels_dict)
    logging.info((N, K))
    client_num = args.num_agents

    min_size = 0
    while min_size < 10:
        idx_batch = [[] for _ in range(client_num)]
        for k in labels_dict:
            idx_k = labels_dict[k]

            # get a list of batch indexes which are belong to label k
            np.random.shuffle(idx_k)
            # using dirichlet distribution to determine the unbalanced proportion for each client (client_num in total)
            # e.g., when client_num = 4, proportions = [0.29543505 0.38414498 0.31998781 0.00043216], sum(proportions) = 1
            proportions = np.random.dirichlet(np.repeat(args.beta, client_num))

            # get the index in idx_k according to the dirichlet distribution
            proportions = np.array([p * (len(idx_j) < N / client_num) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]

            # generate the batch list for each client
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    # distribute data to users
    dict_users = defaultdict(list)
    for user_idx in range(args.num_agents):
        dict_users[user_idx] = idx_batch[user_idx]
        np.random.shuffle(dict_users[user_idx])

    num = [ [ 0 for k in range(K) ] for i in range(client_num)]
    for k in range(K):
        for i in dict_users:
            num[i][k] = len(np.intersect1d(dict_users[i], labels_dict[k]))
    # logging.info(num)
    # print(dict_users)
    # def intersection(lst1, lst2):
    #     lst3 = [value for value in lst1 if value in lst2]
    #     return lst3
    # client_label_num = [len(intersection (dict_users[i], dict_users[i+1] )) for i in range(args.num_agents)]

    for each_client, id_ in zip(num, range(len(num))):
        logging.info('client:%d, distribution: %s' % (id_, each_client))
    return dict_users


def distribute_data(dataset, args, n_classes=10):
    # logging.info(dataset.targets)
    # logging.info(dataset.classes)
    class_per_agent = n_classes

    if args.num_agents == 1:
        return {0: range(len(dataset))}

    def chunker_list(seq, size):
        return [seq[i::size] for i in range(size)]

    # sort labels
    labels_sorted = torch.tensor(dataset.targets).sort()
    # print(labels_sorted)
    # create a list of pairs (index, label), i.e., at index we have an instance of  label
    class_by_labels = list(zip(labels_sorted.values.tolist(), labels_sorted.indices.tolist()))
    # convert list to a dictionary, e.g., at labels_dict[0], we have indexes for class 0
    labels_dict = defaultdict(list)
    for k, v in class_by_labels:
        labels_dict[k].append(v)

    # split indexes to shards
    shard_size = len(dataset) // (args.num_agents * class_per_agent)
    slice_size = (len(dataset) // n_classes) // shard_size
    for k, v in labels_dict.items():
        labels_dict[k] = chunker_list(v, slice_size)
    hey = copy.deepcopy(labels_dict)
    # distribute shards to users
    dict_users = defaultdict(list)
    for user_idx in range(args.num_agents):
        class_ctr = 0
        for j in range(0, n_classes):
            if class_ctr == class_per_agent:
                break
            elif len(labels_dict[j]) > 0:
                dict_users[user_idx] += labels_dict[j][0]
                del labels_dict[j % n_classes][0]
                class_ctr += 1
        np.random.shuffle(dict_users[user_idx])
    # num = [ [ 0 for k in range(n_classes) ] for i in range(args.num_agents)]
    # for k in range(n_classes):
    #     for i in dict_users:
    #         num[i][k] = len(np.intersect1d(dict_users[i], hey[k]))
    # logging.info(num)
    # logging.info(args.num_agents)
    # def intersection(lst1, lst2):
    #     lst3 = [value for value in lst1 if value in lst2]
    #     return lst3
    # logging.info( len(intersection (dict_users[0], dict_users[1] )))

    return dict_users


def get_datasets(data):
    """ returns train and test datasets """
    train_dataset, test_dataset = None, None
    data_dir = './data'

    if data == 'fmnist':
        transform = transforms.Compose([transforms.ToTensor(), 
                                        transforms.Lambda(lambda x: x.to(torch.float64)),
                                        transforms.Normalize(mean=[0.5], std=[0.5])])
        train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True, transform=transform)
    if data == 'mnist':
        transform = transforms.Compose([transforms.ToTensor(), 
                                        transforms.Lambda(lambda x: x.to(torch.float64)),
                                        transforms.Normalize(mean=[0.5], std=[0.5])])
        train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    elif data == 'fedemnist':
        train_dir = '../data/Fed_EMNIST/fed_emnist_all_trainset.pt'
        test_dir = '../data/Fed_EMNIST/fed_emnist_all_valset.pt'
        train_dataset = torch.load(train_dir)
        test_dataset = torch.load(test_dir)

    elif data == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.to(torch.float64)),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.to(torch.float64)),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
        ])
        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform_test)
        train_dataset.targets, test_dataset.targets = torch.LongTensor(train_dataset.targets), torch.LongTensor(
            test_dataset.targets)
    elif data == 'cifar100':
        transform = transforms.Compose([
                                        transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                         transforms.Lambda(lambda x: x.to(torch.float64)),
                                        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                                             std=[0.2675, 0.2565, 0.2761])])
        valid_transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Lambda(lambda x: x.to(torch.float64)),
                                              transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                                                   std=[0.2675, 0.2565, 0.2761])])
        train_dataset = datasets.CIFAR100(data_dir,
                                          train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR100(data_dir,
                                         train=False, download=True, transform=valid_transform)
        train_dataset.targets, test_dataset.targets = torch.LongTensor(train_dataset.targets), torch.LongTensor(
            test_dataset.targets)
    elif data == "tinyimagenet":
        _data_transforms = {
            'train': transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.to(torch.float64))
            ]),
            'test': transforms.Compose([
                transforms.ToTensor(),
                 transforms.Lambda(lambda x: x.to(torch.float64))
            ]),
        }
        _data_dir = './data/tiny-imagenet-200/'
        train_dataset = datasets.ImageFolder(os.path.join(_data_dir, 'train'),
                                             _data_transforms['train'])
        # print(train_dataset[0][0].shape)
        test_dataset = datasets.ImageFolder(os.path.join(_data_dir, 'test'),
                                            _data_transforms['test'])
        train_dataset.targets = torch.tensor(train_dataset.targets)
        test_dataset.targets = torch.tensor(test_dataset.targets)
    return train_dataset, test_dataset


def get_loss_n_accuracy(model, criterion, data_loader, args, round, num_classes=10, sent140_flag=False):
    """ Returns the loss and total accuracy, per class accuracy on the supplied data loader """

    # disable BN stats during inference
    model.eval()
    total_loss, correctly_labeled_samples = 0, 0
    # confusion_matrix = torch.zeros(num_classes, num_classes)
    # not_correct_samples = []
    # forward-pass to get loss and predictions of the current batch
    all_labels = []
    # if round % 20 == 0:
    #     def hook(module, fea_n, fea_out):
    #         representation.append(fea_out.detach().cpu())
    #         return None
    #     for (name, module) in model.named_modules():
    #         if name == layer_name:
    #             handle = module.register_forward_hook(hook=hook)

    for _, (inputs, labels) in enumerate(data_loader):
        inputs, labels = inputs.to(device=args.device, non_blocking=True), \
                         labels.to(device=args.device, non_blocking=True)
        # compute the total loss over minibatch
        if sent140_flag:
            labels = torch.ones_like(labels)
        outputs = model(inputs)
        if args.data == 'sen140':
            avg_minibatch_loss = criterion(outputs.float(), labels.float())
        else:
            avg_minibatch_loss = criterion(outputs, labels)

        total_loss += avg_minibatch_loss.item() * outputs.shape[0]

        # get num of correctly predicted inputs in the current batch
        if args.data == 'sen140':
            # print(outputs.shape)
            # print(outputs)
            pred_labels = outputs.squeeze() > 0.5
            # _, pred_labels = torch.max(outputs, 1)
            # print(pred_labels)
        else:
            _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        all_labels.append(labels.cpu().view(-1))
        # correct_inputs = labels[torch.nonzero(torch.eq(pred_labels, labels) == 0).squeeze()]
        # not_correct_samples.append(  wrong_inputs )
        correctly_labeled_samples += torch.sum(torch.eq(pred_labels, labels)).item()
        # fill confusion_matrix
        # for t, p in zip(labels.view(-1), pred_labels.view(-1)):
            # confusion_matrix[t.long(), p.long()] += 1

    avg_loss = total_loss / len(data_loader.dataset)
    accuracy = correctly_labeled_samples / len(data_loader.dataset)
    # per_class_accuracy = confusion_matrix.diag() / confusion_matrix.sum(1)
    return accuracy

def poison_dataset(dataset, args, data_idxs=None, poison_all=False, agent_idx=-1, modify_label=True):
    # if data_idxs != None:
    #     all_idxs = list(set(all_idxs).intersection(data_idxs))
    if data_idxs != None:
        all_idxs = (dataset.targets != args.target_class).nonzero().flatten().tolist()
        all_idxs = list(set(all_idxs).intersection(data_idxs))
    else:
        all_idxs = (dataset.targets != args.target_class).nonzero().flatten().tolist()
    poison_frac = 1 if poison_all else args.poison_frac
    poison_idxs = random.sample(all_idxs, floor(poison_frac * len(all_idxs)))
    for idx in poison_idxs:
        if args.data == 'fedemnist':
            clean_img = dataset.inputs[idx]
        elif args.data == "tinyimagenet":
            clean_img = dataset[idx][0]
        else:
            clean_img = dataset.data[idx]
        bd_img = add_pattern_bd(clean_img, dataset.targets[idx], args.data, pattern_type=args.pattern_type,
                                agent_idx=agent_idx, attack=args.attack)
        if args.data == 'fedemnist':
            dataset.inputs[idx] = torch.tensor(bd_img)
        elif args.data == "tinyimagenet":
            # don't do anything for tinyimagenet, we poison it in run time
            return
        else:
            dataset.data[idx] = torch.tensor(bd_img)
        if modify_label:
            dataset.targets[idx] = args.target_class
    return poison_idxs


def init_masks(params, sparsities):
    masks = {}
    for name in params:
        masks[name] = torch.zeros_like(params[name])
        dense_numel = int((1 - sparsities[name]) * torch.numel(masks[name]))
        if dense_numel > 0:
            temp = masks[name].view(-1)
            perm = torch.randperm(len(temp))
            perm = perm[:dense_numel]
            temp[perm] = 1
        masks[name] = masks[name].to("cpu")
    return masks

def parameters_to_vector_sorted(model):
    state_dict = model.state_dict()

    return torch.nn.utils.parameters_to_vector([
                copy.deepcopy(model.state_dict()[name])
                for name in model.state_dict()
            ])

def vector_to_model_sorted(vec, model):
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
    # state_dict = model.state_dict()
    # keys = sorted(state_dict.keys())  # same order as flattening
    # pointer = 0
    # for k in keys:
    #     num_param = state_dict[k].numel()
    #     new_data = vec[pointer:pointer + num_param].view_as(state_dict[k])
    #     state_dict[k].copy_(new_data)
    #     pointer += num_param
    # model.load_state_dict(state_dict)
    # return state_dict

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


def calculate_sparsities(args, params, tabu=[], distribution="ERK"):
    spasities = {}
    if distribution == "uniform":
        for name in params:
            if name not in tabu:
                spasities[name] = 1 - args.dense_ratio
            else:
                spasities[name] = 0
    elif distribution == "ERK":
        logging.info('initialize by ERK')
        total_params = 0
        for name in params:
            total_params += params[name].numel()
        is_epsilon_valid = False
        # # The following loop will terminate worst case when all masks are in the
        # custom_sparsity_map. This should probably never happen though, since once
        # we have a single variable or more with the same constant, we have a valid
        # epsilon. Note that for each iteration we add at least one variable to the
        # custom_sparsity_map and therefore this while loop should terminate.
        dense_layers = set()

        density = args.dense_ratio
        while not is_epsilon_valid:
            # We will start with all layers and try to find right epsilon. However if
            # any probablity exceeds 1, we will make that layer dense and repeat the
            # process (finding epsilon) with the non-dense layers.
            # We want the total number of connections to be the same. Let say we have
            # for layers with N_1, ..., N_4 parameters each. Let say after some
            # iterations probability of some dense layers (3, 4) exceeded 1 and
            # therefore we added them to the dense_layers set. Those layers will not
            # scale with erdos_renyi, however we need to count them so that target
            # paratemeter count is achieved. See below.
            # eps * (p_1 * N_1 + p_2 * N_2) + (N_3 + N_4) =
            #    (1 - default_sparsity) * (N_1 + N_2 + N_3 + N_4)
            # eps * (p_1 * N_1 + p_2 * N_2) =
            #    (1 - default_sparsity) * (N_1 + N_2) - default_sparsity * (N_3 + N_4)
            # eps = rhs / (\sum_i p_i * N_i) = rhs / divisor.

            divisor = 0
            rhs = 0
            raw_probabilities = {}
            for name in params:
                if name in tabu or "running" in name or "track" in name :
                    dense_layers.add(name)
                n_param = np.prod(params[name].shape)
                n_zeros = n_param * (1 - density)
                n_ones = n_param * density

                if name in dense_layers:
                    rhs -= n_zeros
                else:
                    rhs += n_ones
                    raw_probabilities[name] = (
                                                      np.sum(params[name].shape) / np.prod(params[name].shape)
                                              ) ** 1
                    divisor += raw_probabilities[name] * n_param
            epsilon = rhs / divisor
            max_prob = np.max(list(raw_probabilities.values()))
            max_prob_one = max_prob * epsilon
            if max_prob_one > 1:
                is_epsilon_valid = False
                for mask_name, mask_raw_prob in raw_probabilities.items():
                    if mask_raw_prob == max_prob:
                        print(f"Sparsity of var:{mask_name} had to be set to 0.")
                        dense_layers.add(mask_name)
            else:
                is_epsilon_valid = True

        # With the valid epsilon, we can set sparsities of the remaning layers.
        for name in params:
            if name in dense_layers:
                spasities[name] = 0
            else:
                spasities[name] = (1 - epsilon * raw_probabilities[name])
    return spasities


def name_param_to_array(param):
    vec = []
    for name in param:
        # Ensure the parameters are located in the same device
        vec.append(param[name].view(-1))
    return torch.cat(vec)


def vector_to_name_param(vec, name_param_map):
    pointer = 0
    for name in name_param_map:
        # The length of the parameter
        num_param = name_param_map[name].numel()
        # Slice the vector, reshape it, and replace the old data of the parameter
        name_param_map[name].data = vec[pointer:pointer + num_param].view_as(name_param_map[name]).data
        # Increment the pointer
        pointer += num_param

    return name_param_map


def add_pattern_bd(x, y, dataset='cifar10', pattern_type='square', agent_idx=-1, attack="DBA"):
    """
    adds a trojan pattern to the image
    """

    # if cifar is selected, we're doing a distributed backdoor attack (i.e., portions of trojan pattern is split between agents, only works for plus)
    if dataset == 'cifar10' or dataset == "cifar100":
        x = np.array(x.squeeze())
        # logging.info(x.shape)
        row = x.shape[0]
        column = x.shape[1]

        if attack == "periodic_trigger":
            for d in range(0, 3):
                for i in range(row):
                    for j in range(column):
                        x[i][j][d] = max(min(x[i][j][d] + 20 * math.sin((2 * math.pi * j * 6) / column), 255), 0)
            # import matplotlib.pyplot as plt
            # plt.imsave("visualization/input_images/backdoor2.png", x)
            # print(y)
            # plt.show()
        else:
            if pattern_type == 'plus':
                start_idx = 5
                size = 6
                if agent_idx == -1:
                    # vertical line
                    # print('xxxxxxx')
                    for d in range(0, 3):
                        for i in range(start_idx, start_idx + size + 1):
                            if d == 2:
                                x[i, start_idx][d] = 0
                            else:
                                x[i, start_idx][d] = 255
                    # horizontal line
                    for d in range(0, 3):
                        for i in range(start_idx - size // 2, start_idx + size // 2 + 1):
                            if d == 2:
                                x[start_idx + size // 2, i][d] = 0
                            else:
                                x[start_idx + size // 2, i][d] = 255
                else:
                    if attack == "DBA":
                        # DBA attack
                        # upper part of vertical
                        if agent_idx % 4 == 0:
                            for d in range(0, 3):
                                for i in range(start_idx, start_idx + (size // 2) + 1):
                                    if d == 2:
                                        x[i, start_idx][d] = 0
                                    else:
                                        x[i, start_idx][d] = 255

                        # lower part of vertical
                        elif agent_idx % 4 == 1:
                            for d in range(0, 3):
                                for i in range(start_idx + (size // 2) + 1, start_idx + size + 1):
                                    if d == 2:
                                        x[i, start_idx][d] = 0
                                    else:
                                        x[i, start_idx][d] = 255

                        # left-part of horizontal
                        elif agent_idx % 4 == 2:
                            for d in range(0, 3):
                                for i in range(start_idx - size // 2, start_idx - size // 4 + 1):
                                    if d == 2:
                                        x[start_idx + size // 2, i][d] = 0
                                    else:
                                        x[start_idx + size // 2, i][d] = 255
                        # right-part of horizontal
                        elif agent_idx % 4 == 3:
                            for d in range(0, 3):
                                for i in range(start_idx - size // 4 + 1, start_idx + size // 2 + 1):
                                    if d == 2:
                                        x[start_idx + size // 2, i][d] = 0
                                    else:
                                        x[start_idx + size // 2, i][d] = 255
                    else:
                        # vertical line
                        for d in range(0, 3):
                            for i in range(start_idx, start_idx + size + 1):
                                if d == 2:
                                    x[i, start_idx][d] = 0
                                else:
                                    x[i, start_idx][d] = 255
                        # horizontal line
                        for d in range(0, 3):
                            for i in range(start_idx - size // 2, start_idx + size // 2 + 1):
                                if d == 2:
                                    x[start_idx + size // 2, i][d] = 0
                                else:
                                    x[start_idx + size // 2, i][d] = 255

                # import matplotlib.pyplot as plt
                #
                # plt.imsave("visualization/input_images/backdoor2.png", x)
                # print(y)
                # plt.show()

    elif dataset == 'tinyimagenet':
        if pattern_type == 'plus':
            start_idx = 5
            size = 6
            # vertical line
            for d in range(0, 3):
                for i in range(start_idx, start_idx + size + 1):
                    if d == 2:
                        x[d][i][start_idx] = 0
                    else:
                        x[d][i][start_idx] = 1
            # horizontal line
            for d in range(0, 3):
                for i in range(start_idx - size // 2, start_idx + size // 2 + 1):
                    if d == 2:
                        x[d][start_idx + size // 2][i] = 0
                    else:
                        x[d][start_idx + size // 2][i] = 1

            # if agent_idx == -1:
            #     # plt.imsave("visualization/input_images/backdoor2.png", x)
            #     print(y)
            #     plt.show()
            # plt.savefig()

    elif dataset == 'fmnist':
        x = np.array(x.squeeze())
        if pattern_type == 'plus':
            start_idx = 5
            size = 6
            if agent_idx == -1:
                # vertical line
                for i in range(start_idx, start_idx + size + 1):
                    x[i, start_idx] = 255
                # horizontal line
                for i in range(start_idx - size // 2, start_idx + size // 2 + 1):
                    x[start_idx + size // 2, i] = 255
            else:
                if attack == "DBA":
                    # DBA attack
                    # upper part of vertical
                    if agent_idx % 4 == 0:
                        for i in range(start_idx, start_idx + (size // 2) + 1):
                            x[i, start_idx] = 255

                    # lower part of vertical
                    elif agent_idx % 4 == 1:
                        for i in range(start_idx + (size // 2) + 1, start_idx + size + 1):
                            x[i, start_idx] = 255

                    # left-part of horizontal
                    elif agent_idx % 4 == 2:
                        for i in range(start_idx - size // 2, start_idx - size // 4 + 1):
                            x[start_idx + size // 2, i] = 255

                    # right-part of horizontal
                    elif agent_idx % 4 == 3:
                        for i in range(start_idx - size // 4 + 1, start_idx + size // 2 + 1):
                            x[start_idx + size // 2, i] = 255
                else:
                    # vertical line
                    for i in range(start_idx, start_idx + size + 1):
                        x[i, start_idx] = 255
                    # horizontal line
                    for i in range(start_idx - size // 2, start_idx + size // 2 + 1):
                        x[start_idx + size // 2, i] = 255

    elif dataset == 'mnist':
        x = np.array(x.squeeze())
        if pattern_type == 'plus':
            start_idx = 1
            size = 2
            if agent_idx == -1:
                # vertical line
                for i in range(start_idx, start_idx + size + 1):
                    x[i, start_idx] = 255
                # horizontal line
                for i in range(start_idx - size // 2, start_idx + size // 2 + 1):
                    x[start_idx + size // 2, i] = 255
            else:
                if attack == "DBA":
                    # DBA attack
                    # upper part of vertical
                    if agent_idx % 4 == 0:
                        for i in range(start_idx, start_idx + (size // 2) + 1):
                            x[i, start_idx] = 255

                    # lower part of vertical
                    elif agent_idx % 4 == 1:
                        for i in range(start_idx + (size // 2) + 1, start_idx + size + 1):
                            x[i, start_idx] = 255

                    # left-part of horizontal
                    elif agent_idx % 4 == 2:
                        for i in range(start_idx - size // 2, start_idx - size // 4 + 1):
                            x[start_idx + size // 2, i] = 255

                    # right-part of horizontal
                    elif agent_idx % 4 == 3:
                        for i in range(start_idx - size // 4 + 1, start_idx + size // 2 + 1):
                            x[start_idx + size // 2, i] = 255
                else:
                    # vertical line
                    for i in range(start_idx, start_idx + size + 1):
                        x[i, start_idx] = 255
                    # horizontal line
                    for i in range(start_idx - size // 2, start_idx + size // 2 + 1):
                        x[start_idx + size // 2, i] = 255
    # import matplotlib.pyplot as plt
    # if agent_idx == -1:
    #     # plt.imsave("visualization/input_images/backdoor2.png", x)
    #     plt.imshow(x)
    #     print(y)
    #     plt.show()
    return x

def extract_last_layer(net, model="vgg9"):
    bias, weight = None, None
    # if model == "vgg9":
    #     for idx, param in enumerate(net.classifier.parameters()):
    #         if idx:
    #             bias = param.data.cpu().numpy()
    #         else:
    #             weight = param.data.cpu().numpy()
    # elif model == "lenet":
    #     for idx, param in enumerate(net.fc2.parameters()):
    #         if idx:
    #             bias = param.data.cpu().numpy()
    #         else:
    #             weight = param.data.cpu().numpy()

    weights = list(net.values())

    weight = weights[-2].cpu().numpy()
    bias = weights[-1].cpu().numpy()

    return bias, weight 

def vector_to_net_dict(vec: torch.Tensor, net_dict) -> None:
    r"""Convert one vector to the net parameters

    Args:
        vec (Tensor): a single vector represents the parameters of a model.
        parameters (Iterable[Tensor]): an iterator of Tensors that are the
            parameters of a model.
    """
    net_dict = net_dict.state_dict()
    # net_dict = {k: net_dict[k] for k in net_dict.keys()}
    pointer = 0
    for param in net_dict.values():
        # The length of the parameter
        num_param = param.numel()
        # Slice the vector, reshape it, and replace the old data of the parameter
        param.data = vec[pointer:pointer + num_param].view_as(param).data

        # Increment the pointer
        pointer += num_param
    return net_dict


def parameters_dict_to_vector_flt(net_dict) -> torch.Tensor:
    vec = []
    for key, param in net_dict.items():
        # print(key, torch.max(param))
        # if key.split('.')[-1] == 'num_batches_tracked':
        #     continue
        vec.append(param.view(-1))
    return torch.cat(vec)



def setup_logging(args):
    """
    Sets up the logging environment and creates necessary directories.
    
    Args:
        args: Arguments object containing logging parameters like non_iid, alpha, data, and aggr.
        
    Returns:
        dir_path: The directory path where logs and backup files are stored.
    """
    log_formatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Clear existing handlers to avoid duplicates
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    
    logPath = "logs"
    time_str = time.strftime("%Y-%m-%d-%H-%M")

    if args.non_iid:
        iid_str = 'noniid(%.1f)' % args.beta
    else:
        iid_str = 'iid'

    args.exp_name = iid_str + '_pr(%.1f)' % args.poison_frac

    if args.exp_name_extra != '':
        args.exp_name += '_%s' % args.exp_name_extra

    fileName = "%s_%s" % (time_str, args.exp_name)

    dir_path = '%s/%s/attack_%s_ar_%.2f/defense_%s/%s/' % (logPath, args.data, args.attack, args.num_corrupt / args.num_agents, args.aggr, fileName)
    file_path = dir_path + 'backup_file/'

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    if not os.path.exists(file_path):
        os.makedirs(file_path)

    backup_file = ['aggregation.py', 'federated.py', 'agent.py']

    for file in backup_file:
        copyfile('./%s' % file, file_path + file)
    
    # Set up file handler for logging
    file_handler = logging.FileHandler(os.path.join(dir_path, f"{fileName}.log"))
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)
    
    # Set up console handler for logging
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)
    
    # Log initial arguments
    logging.info("Arguments:\n" + pprint.pformat(vars(args), indent=4))
    return dir_path
def get_intermediate_Qmk(masks,whole_grads,Watermark,message,k):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    grad_unwater = whole_grads[masks].clone()
    Q_mk = Watermark.intermediate_Qmk(x=grad_unwater,m=message,k=k)
    Q_mk = Q_mk.to(device)
    return Q_mk
def embedding_on_mask_with_Qmk(masks,whole_grads,Watermark,alpha,Q_mk,quanti_factor=None,model=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    grad_unwater = whole_grads[masks].clone()
    w_ = Watermark.embed_with_Qmk(grad_unwater, alpha=alpha, q_mk=Q_mk, quanti_factor=quanti_factor)
    whole_grads[masks] = w_.to(device)
    return whole_grads

def embedding_watermark_on_position(masks,whole_grads,Watermark,message,alpha,k,quanti_factor=None,model=None):
    # device = whole_grads.device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # alpha = args.alpha
    # k = args.k
    # delta = args.delta
    # print('Alpha used in embedding: ', alpha, delta, k)
    # print(whole_grads.dtype, type(whole_grads))
    # Extract the section to watermark
    # grad_unwater = copy.deepcopy(whole_grads[masks[0]:masks[1]])
    
    grad_unwater = whole_grads[masks[0]:masks[1]].clone()
    # grad_unwater = grad_unwater.cpu().detach().numpy()
    if len(masks) !=2:
        grad_unwater = whole_grads[masks].clone()
    # grad_unwater = grad_unwater.detach().cpu()
    # print('type alpha:', type(alpha))
    # print('type k:', type(k))
    # print(grad_unwater.dtype, type(grad_unwater))
    # if type(alpha) != float:
    #     # print('type alpha:', type(alpha))
    #     alpha = alpha.cpu().numpy()
    # if type(k) != float and type(k) != int:
    #     print('type k:', type(k))
        # k = k.cpu().numpy()
    w_ = Watermark.embed(grad_unwater, m=message, alpha=alpha, k=k, quanti_factor=quanti_factor)
    # w_ = torch.tensor(w_,dtype=whole_grads.dtype).to(device)
    w_ = w_.to(device)
    # Update the flat tensor
    # whole_grads[masks[0]:masks[1]].copy_(w_)
    if len(masks) !=2:
        # whole_grads[masks].copy_(w_)
        whole_grads[masks] = w_
    else:
        whole_grads[masks[0]:masks[1]].copy_(w_)
    # If model is provided, update the actual model parameters in-place
    if model is not None:
        with torch.no_grad():
            # vector_to_parameters(whole_grads,model.parameters())
            vector_to_model_sorted(whole_grads, model)
            # start = 0
            # for p in model.parameters():
            #     numel = p.numel()
            #     p.data.copy_(whole_grads[start:start+numel].view_as(p))
            #     start += numel
    return whole_grads
def extract_on_position(masks,whole_grads,Watermark,k):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    grad_water = whole_grads[masks].clone()
    message,dm_hat = Watermark.extract(grad_water,k=k)
    # r_w = torch.tensor(r_w,dtype=whole_grads.dtype)
    return message,dm_hat
def recover_on_position(masks,whole_grads,Watermark,alpha,dm_hat=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    grad_water = whole_grads[masks].clone()
    r_w = Watermark.recover(grad_water,alpha=alpha,dm_hat=dm_hat)
    reconstructed_grad = r_w.to(device)
    whole_grads[masks]= reconstructed_grad
    return whole_grads
def detect_recover_on_position(masks,whole_grads,Watermark,alpha,k,quanti_factor=None,model=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # alpha = args.alpha
    # k = args.k
    # delta = args.delta
    # print('Alpha used in detecting: ',alpha,delta,k)
    # print(whole_grads.dtype, type(whole_grads))
    grad_water = whole_grads[masks[0]:masks[1]].clone()
    if len(masks) !=2:
        grad_water = whole_grads[masks].clone()
    # grad_water = grad_water.detach().cpu()
    # grad_water = grad_water.cpu().detach().numpy()
    # print(grad_water.dtype, type(grad_water))
    # if type(alpha) != float:
    #     alpha = alpha.cpu().numpy()
    # if type(k) != float and type(k) != int:
    #     k = k.cpu().numpy()
    r_w,mm = Watermark.detect(grad_water,alpha=alpha,k=k,quanti_factor=quanti_factor)
    # r_w = torch.tensor(r_w,dtype=whole_grads.dtype)
    # mm = torch.tensor(mm,dtype=torch.int)
    reconstructed_grad = r_w.to(device)
    if len(masks) !=2:
        whole_grads[masks]= reconstructed_grad
    else:
        whole_grads[masks[0]:masks[1]].copy_(reconstructed_grad)

    if model is not None:
        with torch.no_grad():
            # vector_to_parameters(whole_grads,model.parameters())
            vector_to_model_sorted(whole_grads, model)
            # start = 0
            # for p in model.parameters():
            #     numel = p.numel()
            #     p.data.copy_(whole_grads[start:start+numel].view_as(p))
            #     start += numel
    return whole_grads, mm


def get_mean_updates(updates,avg_update=None):
    distances_matrix = []
    if avg_update is None:
        avg_update = copy.deepcopy(updates[0])
        for i in updates[1:]:
            avg_update += i
        avg_update = avg_update / len(updates)
    for idx, w_local in enumerate(updates):
        distance = euclidean_distance(w_local, avg_update)
        distances_matrix.append(distance)

    distances = distances_matrix
    # distances_array = np.array(distances).reshape(-1, 1)
    return 

    
def euclidean_distance(local_weights, global_weights):
    distance = 0
    # for key in global_weights.keys():
    #     distance += torch.pow(local_weights[key] - global_weights[key], 2).sum()
    distance = torch.sum((local_weights - global_weights) ** 2)
    distance = torch.sqrt(distance)
    return distance.item()

def decompose_normal_distributions(data, n_components=2):
    '''
    Decompose the data into two normal distributions using Gaussian Mixture Model (GMM).
    Returns the data points in the largest cluster, bounds of the cluster, means, covariances, and weights of the GMM.
    '''
    from sklearn.mixture import GaussianMixture
    gmm = GaussianMixture(n_components=n_components, random_state=0)
    gmm.fit(data.reshape(-1, 1))
    labels = gmm.predict(data.reshape(-1, 1))
    unique_labels, counts = np.unique(labels, return_counts=True)
    max_cluster_index = unique_labels[np.argmax(counts)]
    max_cluster_data = data[labels == max_cluster_index]
    bounds = (max_cluster_data.min(), max_cluster_data.max())
    return max_cluster_data, bounds, gmm.means_, gmm.covariances_, gmm.weights_

def plot_control_chart(client_id, client_means, distances_matrix, save_dir, L=3, attack=None):
    """
    SPC-based anomaly detection algorithm.
    """
    from matplotlib import pyplot as plt
    ano = []
    distances = [distance for client_list in distances_matrix for distance in client_list]
    std = np.std(distances)
    mean = np.mean(distances)
    # the control limit to select clients, in our study LCL is not used
    UCL = mean + L *std # Upper Control Limit, sort like the upper bound
    LCL = mean - L *std
    
    plt.figure(figsize=(10, 6))
    plt.plot(client_id, client_means, marker='o', linestyle='-', color='blue', label='Average Distance')
    # if the client weight mean is larger than the upper control limit, mark it as anomaly
    for idx, client_mean in zip(client_id, client_means):
        if client_mean > UCL:
            ano.append(idx)
            plt.plot(idx, client_mean, marker='o', color='red')

    plt.axhline(UCL, color='red', linestyle='--', label='UCL')
    plt.title('Control Chart for All Clients')
    plt.xlabel('Client ID')
    plt.ylabel('Average Distance')
    plt.legend()
    plt.savefig(os.path.join(save_dir, f'control_chart_all_clients.png'))
    plt.close()

    return ano, UCL, LCL

def calculate_accuracy(detected_noisy_clients, actual_noisy_clients):
    """
    Calculate the defense metrix of the anomaly detection algorithm.
    """

    detected_set = set(detected_noisy_clients)
    actual_set = set(actual_noisy_clients)

    correct_detections = detected_set.intersection(actual_set)

    # Calculate recall
    if len(actual_set) > 0:
        R = len(correct_detections) / len(actual_set)
    else:
        R = 0.0 

    # Calculate precision
    if len(detected_set) > 0:
        P = len(correct_detections) / len(detected_set)
    else:
        P = 0.0  

    return R,P

def calculate_alpha(updates,args):
    if args.cal_alpha == 'mean':
        return abs(updates.mean().item())
    elif args.cal_alpha == 'median':
        return abs(updates.median().item())
    elif args.cal_alpha == 'norm':
        return abs(updates.norm().item())
    elif args.cal_alpha == 'min':
        return updates.min().item()
    
def alignIns_alpha_quanti(local_updates, flat_global_model,args):
    '''
    AlignIns defense with MZ-score based anomaly detection.
    1. TDA: Cosine similarity between each local update and the global model.
    2. MPSA: Matching Proportion of Sign Agreement between each local update and the major sign of local updates.
    Return the mean of benign updates after filtering out anomalies.
    '''
    num_chosen_clients = len(local_updates)
    inter_model_updates = torch.stack(local_updates, dim=0)

    tda_list = []
    mpsa_list = []
    major_sign = torch.sign(flat_global_model)
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    for i in range(len(inter_model_updates)):
        # different mask for each local update
        _, init_indices = torch.topk(torch.abs(inter_model_updates[i]), int(len(inter_model_updates[i]) * args.sparsity))
        # calculate MPSA, the matching proportion of sign agreement, 0-1
        mpsa_list.append((torch.sum(torch.sign(inter_model_updates[i][init_indices]) == major_sign[init_indices]) / torch.numel(inter_model_updates[i][init_indices])).item())
        # calculate TDA, the cosine similarity between each local update and (previous round) global model parameters? or update?
        tda_list.append(cos(inter_model_updates[i], flat_global_model).item())


    ######## MZ-score calculation ########
    mpsa_std = np.std(mpsa_list)
    mpsa_med = np.median(mpsa_list)
    # normalized z-score, the smaller the better
    mzscore_mpsa = []
    for i in range(len(mpsa_list)):
        mzscore_mpsa.append(np.abs(mpsa_list[i] - mpsa_med) / mpsa_std)
    print(f"alignIns sign alignment:",mpsa_list)
    tda_std = np.std(tda_list)
    tda_med = np.median(tda_list)
    mzscore_tda = []
    for i in range(len(tda_list)):
        mzscore_tda.append(np.abs(tda_list[i] - tda_med) / tda_std)

    ######## Anomaly detection with MZ score ########

    benign_idx1 = set([i for i in range(num_chosen_clients)])
    # filter with MPSA mzscore, lower than threshold is benign
    benign_idx1 = benign_idx1.intersection(set([int(i) for i in np.argwhere(np.array(mzscore_mpsa) < args.lambda_s)]))
    benign_idx2 = set([i for i in range(num_chosen_clients)])
    benign_idx2 = benign_idx2.intersection(set([int(i) for i in np.argwhere(np.array(mzscore_tda) < args.lambda_c)]))

    benign_set = benign_idx2.intersection(benign_idx1)
    print('Alignins cosine: ',benign_idx2, 'sign:',benign_idx1)
    benign_idx = list(benign_set)
    if len(benign_idx) == 0:
        return torch.zeros_like(local_updates[0])

    benign_updates = torch.stack([local_updates[i] for i in benign_idx], dim=0)

    # mean_benign_update = torch.mean(benign_updates, dim=0)
    # std = torch.std(benign_updates, dim=0)
    print(f"AlignIns mpsa med: {mpsa_med}, mpsa_std: {mpsa_std}")
    return mpsa_med+tda_med, mpsa_std+tda_std, benign_idx

def cal_mzscore(update,flat_global_model,global_update,args):
    '''
    Calculate the combined MZ-score of TDA and MPSA for a single update.
    1. TDA: Cosine similarity between the update and the global model.
    2. MPSA: Matching Proportion of Sign Agreement between the update and the global model.
    Return the sum of TDA and MPSA as the anomaly
    '''
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    tda = cos(update, flat_global_model).item()
    # tda_std = args.tda_std
    # tda_med = args.tda_med
    # mzscore_tda = np.abs(tda - tda_med) / tda_std
    major_sign = torch.sign(global_update)
    _, init_indices = torch.topk(torch.abs(update), int(len(update) * args.sparsity))
    # print('init indices length:', len(init_indices))
    # print(len(update[init_indices]))
    mpsa = torch.sum(torch.sign(update[init_indices]) == major_sign[init_indices]) / torch.numel(update[init_indices])
    # mpsa_std = args.mpsa_std
    # mpsa_med = args.mpsa_med
    # mzscore_mpsa = np.abs(mpsa - mpsa_med) / mpsa_std

    return mpsa+tda




import numpy as np
import hashlib

class CosineFuzzyExtractor:
    def __init__(self, input_dim, projection_count=255, error_tolerance=0.10):
        """
        input_dim: Dimension of the model update vector (e.g., 10,000)
        projection_count: Length of the discrete key/hash (security parameter)
        error_tolerance: Max % of bit differences allowed (10% diff approx 18 degree angle)
        """
        self.input_dim = input_dim
        self.projection_count = projection_count
        self.threshold = int(projection_count * error_tolerance)
        
        # In a real FL system, these 'projections' act as the public randomness 
        # shared between Server and Clients. They define the "Hyperplanes".
        # We seed it so it's deterministic for everyone.
        np.random.seed(42) 
        self.projections = np.random.randn(projection_count, input_dim)

    def _quantize(self, vector):
        """
        SimHash: Converts continuous vector to binary string based on Cosine Similarity.
        1. Project vector onto random hyperplanes.
        2. Take the sign (+ or -).
        3. Convert to bits.
        """
        # Dot product captures the angle information
        projected = np.dot(self.projections, vector)
        # Convert sign to bits (Positive -> 1, Negative -> 0)
        bits = (projected > 0).astype(int)
        return bits

    def gen(self, w_ref):
        """
        GENERATE (Server Side):
        Input: The clean Reference Update (w_ref)
        Output: (Key, Helper_String)
        """
        # 1. Get the "Bio-features" (The SimHash bits of the reference)
        w_bits = self._quantize(w_ref)
        
        # 2. Create a Random Secret Key R
        # In a real implementation, we would encode R using BCH code to get C.
        # Here, we treat w_bits as the "noiseless codeword" for simulation.
        key_R = hashlib.sha256(w_bits.tobytes()).hexdigest()
        
        # 3. Create Helper String P (Secure Sketch)
        # In Code-Offset construction: P = C XOR w_bits.
        # Since we assume w_bits IS the codeword for this demo: P = 0 (or masked).
        # For this simulation, the "Helper" is the instruction to use the specific projections.
        
        return key_R, w_bits

    def rep(self, w_client, w_ref_bits):
        """
        REPRODUCE (Client Side):
        Input: Noisy Client Update (w_client), Server's Reference Bits (Helper)
        Output: Recovered Key (or None if failed)
        """
        # 1. Quantize the client's continuous vector
        client_bits = self._quantize(w_client)
        
        # 2. Calculate Hamming Distance (Bit Errors)
        # XOR to find differences, then sum
        errors = np.sum(client_bits != w_ref_bits)
        error_rate = errors / self.projection_count
        
        print(f"DEBUG: Bit Error Rate: {error_rate:.2%} (Threshold: {self.threshold/self.projection_count:.2%})")

        # 3. Error Correction (Simulated)
        # If errors < threshold, ECC would succeed in correcting client_bits to match w_ref_bits.
        if errors <= self.threshold:
            # Success! The ECC corrects the errors, recovering the original reference bits
            recovered_bits = w_ref_bits 
            recovered_key = hashlib.sha256(recovered_bits.tobytes()).hexdigest()
            return recovered_key
        else:
            # Fail! Too much noise/poison. ECC fails to decode.
            return None

# # --- SIMULATION ---

# # 1. Setup Environment
# dim = 1000 # Dimension of model update
# extractor = CosineFuzzyExtractor(input_dim=dim, projection_count=1024, error_tolerance=0.15)

# # 2. Create Data
# # Server Reference Update (Random vector)
# w_ref = np.random.randn(dim) 

# # Benign Client (High Cosine Similarity)
# # We add small noise, preserving direction
# noise = np.random.randn(dim) * 0.5 
# w_benign = w_ref + noise 

# # Malicious Client (Different Direction/Angle)
# # We create a vector orthogonal or opposing the reference
# w_malicious = np.random.randn(dim) # Random vector is likely orthogonal in high dim

# # --- EXECUTION ---

# print("--- SERVER: Generating Lock ---")
# secret_key, public_helper = extractor.gen(w_ref)
# print(f"Secret Key Generated: {secret_key[:10]}...")

# print("\n--- CLIENT A (Benign): Attempting to Unlock ---")
# # Calculate actual Cosine Similarity for verification
# cos_sim_benign = np.dot(w_ref, w_benign) / (np.linalg.norm(w_ref) * np.linalg.norm(w_benign))
# print(f"Actual Cosine Similarity: {cos_sim_benign:.4f}")

# key_benign = extractor.rep(w_benign, public_helper)
# if key_benign == secret_key:
#     print(">> ACCESS GRANTED: Keys Match!")
# else:
#     print(">> ACCESS DENIED.")

# print("\n--- CLIENT B (Malicious): Attempting to Unlock ---")
# # Calculate actual Cosine Similarity
# cos_sim_mal = np.dot(w_ref, w_malicious) / (np.linalg.norm(w_ref) * np.linalg.norm(w_malicious))
# print(f"Actual Cosine Similarity: {cos_sim_mal:.4f}")

# key_mal = extractor.rep(w_malicious, public_helper)
# if key_mal == secret_key:
#     print(">> ACCESS GRANTED: Keys Match!")
# else:
#     print(">> ACCESS DENIED: Cannot reproduce key.")

def num_not_same(tensor_a, tensor_b):
    return torch.sum(tensor_a != tensor_b).item()
def compute_performance_drop(watered_host, original_model, criterion, val_loader, args, val_acc):
    # print(torch.sum(abs((host-watered_host))), torch.allclose(host, watered_host))
    # exclude_indices = torch.ones(len(host), dtype=bool)
    # exclude_indices[mask] = False
    # print(torch.allclose(host[exclude_indices], watered_host[exclude_indices]))
    # print(torch.allclose(host[mask], watered_host[mask]))
    watered_model = copy.deepcopy(original_model)
    vector_to_parameters(watered_host, watered_model.parameters())
    watered_val_acc = get_loss_n_accuracy(
                        watered_model, criterion, val_loader, args, 0, 0
                    )
    print(f"Watermarked model validation accuracy: {watered_val_acc*100:.2f}%, drop: {val_acc - watered_val_acc:.4f}")

def compare_hashes(hash1, hash2):
    matches = 0
    for h1, h2 in zip(hash1, hash2):
        for i in range(len(h1)):
            if h1[i] == h2[i]:
                matches += 1
    return matches

def get_hashes(lsh, input_point):
    return [lsh._hash(planes, input_point) for planes in lsh.uniform_planes]
def bump(alpha):
    return torch.exp(-(alpha-0.5)**2 / 10**2)
def sigmoid(alpha):
    return 1/(1+torch.exp(-alpha))
def quasi_periodic(t):
    if not isinstance(t,torch.Tensor):
        t = torch.tensor(t).to(torch.float64)
    return (((torch.sin(t) + torch.sin(math.sqrt(2) * t))+2)/4)**1.1
def logistic_map(x, r=3.6):
    return r * x * (1 - x)
def sine_map(x,r=3.9):
    return r * torch.sin(torch.pi * x)
def CTBCS(t,f1=logistic_map,f2=sine_map,beta=0.5):
    return torch.cos(torch.pi*f1(t)+f2(t,r=1-3.6/4)-beta)

def chaotic_sequence(length, init_value):
    chaotic_alpha = []
    for _ in range(length):
        init_value = CTBCS(init_value,beta=0.5)
        chaotic_alpha.append(init_value)
    return torch.tensor(chaotic_alpha)
    # chaotic_alpha = torch.tensor(chaotic_alpha,dtype=torch.float64).to(args.device).view(args.lsh_dim,args.lsh_size)


import numpy as np
import hashlib

# def hash_to_vector(hash_bits, D=1_971_936):
def hash_to_seed(hash_bits):
    # Convert bits to bytes
    bits_as_bytes = hash_bits.cpu().byte().numpy().tobytes()
    seed_bytes = hashlib.blake2b(bits_as_bytes, digest_size=8).digest()
    return int.from_bytes(seed_bytes, 'little')
# def hash_to_vector_trig(hash_bits, D=1_971_936):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hash_bits = torch.tensor(hash_bits,dtype=torch.float32).to(device)
    t = torch.dot(hash_bits, torch.arange(1, len(hash_bits)+1,dtype=torch.float32).to(device)) % (2*torch.pi)

    idx = torch.arange(D).to(device)
    v = torch.sin(t * idx).to(device) + torch.cos((t+1) * idx).to(device)
    return torch.tensor(v)
# def hash_mz(major_sign, list_sign, lambda_s):
#     mpsa_list = []
    
#     for i in range(list_sign.shape[0]):
#         mpsa_list.append(torch.sum(major_sign != list_sign[i]).item()/len(major_sign))
#     # lsh_dis_np = np.array(mpsa_list).reshape(-1, 1)
#     # largest_cluster_data, bounds, means, covariances, weights = decompose_normal_distributions(lsh_dis_np)
#     # benign_clients = []
#     # for i in range(list_sign.shape[0]):
#     #     if mpsa_list[i] in largest_cluster_data:
#     #         benign_clients.append(i)
#     # print(f"Using clustering, the benginclients are: {benign_clients}")
#     mpsa_std = np.std(mpsa_list)
#     mpsa_med = np.median(mpsa_list)
#     # normalized z-score, the smaller the better
#     mzscore_mpsa = []
#     for i in range(len(mpsa_list)):
#         mzscore_mpsa.append(np.abs(mpsa_list[i] - mpsa_med) / mpsa_std)
#     # if mpsa_std == 0:
#     #     mzscore_mpsa = np.zeros_like(mpsa_list_np)
#     # else:
#     #     mzscore_mpsa = np.abs(mpsa_list_np - mpsa_med) / mpsa_std
#     # if euclid:
#     #     eucllid_list = []
#     #     for i in range()
#     benign_idx = [int(i) for i in np.argwhere(np.array(mzscore_mpsa) < lambda_s)]
#     return benign_idx,mzscore_mpsa

def hash_mz(major_sign, list_sign, lambda_s):
    # 1. Vectorized Hamming Distance calculation (NO LOOP)
    # major_sign: [D] -> [1, D]
    # list_sign:  [N, D]
    # (major_sign != list_sign) results in [N, D] boolean tensor
    # .sum(dim=1) gives [N] counts
    if major_sign.dim() == 1:
        major_sign = major_sign.unsqueeze(0)
    mpsa_tensor = torch.sum(major_sign != list_sign, dim=1).float() / major_sign.size(0)
    
    # Move to CPU once for the statistical libraries
    # mpsa_list_np = mpsa_tensor.cpu().numpy()

    # 4. Optimized Z-Score
    mpsa_std = torch.std(mpsa_tensor)
    mpsa_med = torch.median(mpsa_tensor)
    
    # Avoid zero-division if all clients are identical
    if mpsa_std == 0:
        mzscore_mpsa = torch.zeros_like(mpsa_tensor)
    else:
        mzscore_mpsa = torch.abs(mpsa_tensor - mpsa_med) / mpsa_std
    
    # 5. Fast index extraction
    benign_idx = torch.where(mzscore_mpsa < lambda_s)[0].cpu().tolist()

    return benign_idx, mzscore_mpsa
def hash_euclid(ref_update, local_updates):
    euclid_dis = []
    for i in range(len(local_updates)):
        euclid_dis.append(euclidean_distance(local_updates[i],ref_update))
    euclid_dis_np = np.array(euclid_dis).reshape(-1, 1)
    largest_cluster_data, bounds, means, covariances, weights = decompose_normal_distributions(euclid_dis_np)
    benign_clients = []
    for i in range(len(local_updates)):
        if euclid_dis[i] in largest_cluster_data:
            benign_clients.append(i)
    return benign_clients

def meature_metrics(benign_list,args):
    print()
    # from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
    # import torch

    # # 1. Prepare Ground Truth (y_true)
    # # Assuming indices 0 to args.num_corrupt-1 are malicious (Label 1)
    # # and the rest are benign (Label 0)
    # total_clients = args.num_agents # Adjust based on your actual total
    # y_true = [1 if i < args.num_corrupt else 0 for i in range(total_clients)]

    # # 2. Prepare Predictions (y_pred)
    # # Based on your logic: if score < threshold, it's flagged
    # # (Replace 'threshold' with your actual filtering cutoff)
    # scores = torch.tensor(benign_list)
    # y_pred = (scores < args.num_corrupt).int().tolist() 

    # # 3. Calculate Metrics
    # precision = precision_score(y_true, y_pred)
    # recall = recall_score(y_true, y_pred)
    # f1 = f1_score(y_true, y_pred)
    # auroc = roc_auc_score(y_true, scores) # Use raw scores for AUROC

    # print(f"Precision: {precision:.4f}")
    # print(f"Recall (Detection): {recall:.4f}")
    # print(f"F1-Score: {f1:.4f}")
    # print(f"AUROC: {auroc:.4f}")
def free_memory(*vars):
    import gc, torch
    for var in vars:
        del var
    gc.collect()
    torch.cuda.empty_cache()

# def generat_params(seed,length,param_length):

#     # seed = hash_to_seed([seed])
#     generator = torch.Generator().manual_seed(seed)
#     # seed = 
#     beta_id = torch.bernoulli(torch.full((length,), 0.5),generator=generator).int()
#     delta = torch.rand(length, generator=generator)
#     # random_idx = torch.randint(0, len(delta), (1,))
#     # delta[random_idx] = 1e-7
#     alpha = torch.clamp(torch.randn(length,generator=generator).double() * 0.49 + 0.5, 0.5+1e-4, 1-1e-4)
#     k_out = torch.rand(length,generator=generator)
#     # masks = random.sample(range(param_length),length, generator=generator)
#     rng = np.random.default_rng(seed=seed)
#     masks = list(rng.choice(range(param_length), size=length, replace=False))

#     return beta_id,delta,alpha,k_out, masks
def generat_params(seed, length, param_length, device='cuda'):
    # 1. Initialize Generator on the specific device to avoid CPU-GPU sync
    generator = torch.Generator(device=device).manual_seed(seed)
    
    beta_id = torch.bernoulli(torch.full((length,), 0.5, device=device), generator=generator).int()
    delta = torch.rand(length, generator=generator, device=device)
    alpha = torch.randn(length, generator=generator, device=device).double() * 0.49 + 0.5
    alpha = torch.clamp(alpha, 0.501, 0.778)
    
    k_out = torch.rand(length, generator=generator, device=device)
    # msk_gen = torch.Generator(device=device).manual_seed(0)
    masks = torch.randperm(param_length, generator=generator, device=device)[:length]
    
    return beta_id, delta, alpha, k_out, masks
# def inital_secret():
#     import secrets
#     import string
#     alphabet = (
#     string.ascii_letters +
#     string.digits +
#     "!@#$%^&*()-_=+"
#     )

#     password = ''.join(secrets.choice(alphabet) for _ in range(12))
#     return password
def initial_secret():
    import secrets
    # Generate a 256-bit seed (more standard for modern security)
    seed_256bit = secrets.token_hex(32) # 32 bytes = 256 bits
    return seed_256bit

import hashlib

def generate_round_seed(master_seed, round_number):
    """
    Generates a unique 128-bit seed for a specific round.
    """
    # Combine the seed and round into a single string
    input_str = f"{master_seed}-{round_number}".encode()
    
    hash_digest = hashlib.sha256(input_str).digest()
    
    # We take 8 bytes (64 bits) which is safe for most systems
    seed_int = int.from_bytes(hash_digest[:8], byteorder='big')
    
    # 3. Optional: Fit it into a 32-bit space if using older libraries
    # seed_int = seed_int % (2**32)
    
    return seed_int

if __name__ == "__main__":
    for i in range(3):
        print(generat_params(73,5,10))