import utils
import models
import math
import copy
import numpy as np
from agent import Agent
from agent_sparse import Agent as Agent_s
from aggregation import Aggregation
import torch
import random
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.nn.utils import parameters_to_vector
import logging
import argparse
import os
import warnings
from watermarks.modi_qim import QIM
from attacks import attack
import datetime
# from lshashpy3 import LSHash
from LshCls import LSH
from collections import defaultdict
import time
warnings.filterwarnings("ignore")
date = datetime.datetime.now().strftime("%Y%m%d")
if __name__ == "__main__":
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.set_default_dtype(torch.float64)
    # torch.use_deterministic_algorithms(True)
    parser = argparse.ArgumentParser(description="pass in a parameter")

    parser.add_argument(
        "--data", type=str, default="cifar10", help="dataset we want to train on"
    )
    parser.add_argument("--num_agents", type=int, default=20, help="number of agents:K")
    parser.add_argument(
        "--agent_frac", type=float, default=1.0, help="fraction of agents per round:C"
    )
    parser.add_argument(
        "--num_corrupt", type=int, default=2, help="number of corrupt agents"
    )
    parser.add_argument(
        "--rounds", type=int, default=150, help="number of communication rounds:R"
    )
    parser.add_argument(
        "--local_ep", type=int, default=2, help="number of local epochs:E"
    )
    parser.add_argument("--bs", type=int, default=64, help="local batch size: B")
    parser.add_argument(
        "--client_lr", type=float, default=0.1, help="clients learning rate"
    )
    parser.add_argument(
        "--server_lr", type=float, default=1., help="servers learning rate"
    )
    parser.add_argument(
        "--target_class", type=int, default=7, help="target class for backdoor attack"
    )
    parser.add_argument(
        "--poison_frac",
        type=float,
        default=0.5,
        help="fraction of dataset to corrupt for backdoor attack",
    )
    parser.add_argument(
        "--pattern_type", type=str, default="plus", help="shape of bd pattern"
    )
    parser.add_argument(
        "--theta", type=int, default=8, help="break ties when votes sum to 0"
    )
    parser.add_argument(
        "--theta_ld", type=int, default=10, help="break ties when votes sum to 0"
    )
    parser.add_argument(
        "--snap", type=int, default=1, help="do inference in every num of snap rounds"
    )
    parser.add_argument(
        "--device",
        default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        help="To use cuda, set to a specific GPU ID.",
    )
    parser.add_argument(
        "--num_workers", type=int, default=0, help="num of workers for multithreading"
    )
    parser.add_argument(
        "--dense_ratio",
        type=float,
        default=0.25,
        help="num of workers for multithreading",
    )
    parser.add_argument(
        "--anneal_factor",
        type=float,
        default=0.0001,
        help="num of workers for multithreading",
    )
    parser.add_argument(
        "--se_threshold",
        type=float,
        default=1e-4,
        help="num of workers for multithreading",
    )
    parser.add_argument("--non_iid", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument(
        "--attack",
        type=str,
        default="badnet",
        choices=["badnet", "DBA", "neurotoxin", "pgd"],
    )
    parser.add_argument(
        "--aggr",
        type=str,
        default="avg",
        choices=[
            "avg",
            "alignins",
            "rlr",
            "mkrum",
            "mmetric",
            "lockdown",
            "foolsgold",
            "signguard",
            "rfa",
            "flgmm",
        ],
        help="aggregation function to aggregate agents' local weights",
    )
    parser.add_argument("--lr_decay", type=float, default=0.99)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--mask_init", type=str, default="ERK")
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--same_mask", type=int, default=1)
    parser.add_argument("--cease_poison", type=float, default=100000)
    parser.add_argument("--exp_name_extra", type=str, help="defence name", default="")
    parser.add_argument("--super_power", action="store_true")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--sparsity", type=float, default=0.3)
    parser.add_argument("--lambda_s", type=float, default=1.0)
    parser.add_argument("--lambda_c", type=float, default=1.0)
    parser.add_argument("--watermark", action="store_true",help='ordinary watermark (one layer)')
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--delta", type=float, default=1.0)
    parser.add_argument("--k", type=float, default=0)
    parser.add_argument("--job", type=str,default="ALTestDefault")
    parser.add_argument("--backdoor", action="store_true")
    parser.add_argument("--byz", action="store_true")
    parser.add_argument("--byz_attack", default="min_max", choices=["random", "sign_flip", "zero", "noise", "nan", "label_flip", "lie", "byzMean", "min_max", "min_sum", "adaptive_std", "adaptive_sign", "adaptive_uv", "non"], help="the attack method of byzantine agents")
    parser.add_argument("--num_byz", type=int, default=2, help="the number of byzantine agents")
    parser.add_argument("--use_g0", action="store_true")
    # parser.add_argument("--water_num", type=int, default=400, help="the number of watermarks to be embedded")
    parser.add_argument("--water_por", type=float, default=0.03, help="the portion of watermarks to be embedded")
    parser.add_argument("--lsh_filter", action="store_true", help="LSH defence, used alone for LSH only, with watermark & authentication for whole")
    parser.add_argument("--lsh_por", type=float, default=0.3, help="the portion of LSH tested")
    parser.add_argument("--lsh_size", type=int, default=100, help="the portion of LSH tested")
    parser.add_argument("--num_hash_tables", type=int, default=20, help="the portion of LSH tested")
    parser.add_argument("--authentication", action="store_true", help="ordinary watermark + authentication layer (2 layers)")
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    # args.lsh_size = 50
    # args.num_hash_tables = 10
    args = parser.parse_args()

    if args.clean:
        args.num_corrupt = 0
        args.byz = False
        args.num_byz = 0
        args.exp_name_extra = "clean"
    args.logging = logging
    if args.super_power:
        args.exp_name_extra = "sp"
    if args.watermark:
        logging.info("Watermarking")
        args.rqim = QIM(args.delta)
        args.exp_name_extra = args.exp_name_extra + "_wm_a{:.2f}_d{:.2f}_k{:.2f}".format(args.alpha,args.delta,args.k)
    per_data_dict = {
        "rounds": {"fmnist": 50, "cifar10": 150, "cifar100": 100, "tinyimagenet": 50},
        "num_target": {"fmnist": 10, "cifar10": 10, "cifar100": 100, "tinyimagenet": 200,},
    }

    args.rounds = per_data_dict["rounds"][args.data]
    args.num_target = per_data_dict["num_target"][args.data]

    args.log_dir = utils.setup_logging(args)

    train_dataset, val_dataset = utils.get_datasets(args.data)
    backdoor_train_dataset = None
    Attack  = attack(args.byz_attack)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.bs,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    if args.non_iid:
        user_groups = utils.distribute_data_dirichlet(train_dataset, args)
    else:
        user_groups = utils.distribute_data(
            train_dataset, args, n_classes=args.num_target
        )

    idxs = (val_dataset.targets != args.target_class).nonzero().flatten().tolist()

    if args.data != "tinyimagenet":
        poisoned_val_set = utils.DatasetSplit(copy.deepcopy(val_dataset), idxs)
        utils.poison_dataset(poisoned_val_set.dataset, args, idxs, poison_all=True)
    else:
        poisoned_val_set = utils.DatasetSplit(
            copy.deepcopy(val_dataset), idxs, runtime_poison=True, args=args
        )

    poisoned_val_loader = DataLoader(
        poisoned_val_set,
        batch_size=args.bs,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    if args.data != "tinyimagenet":
        idxs = (val_dataset.targets != args.target_class).nonzero().flatten().tolist()
        poisoned_val_set_only_x = utils.DatasetSplit(copy.deepcopy(val_dataset), idxs)
        utils.poison_dataset(
            poisoned_val_set_only_x.dataset,
            args,
            idxs,
            poison_all=True,
            modify_label=False,
        )
    else:
        poisoned_val_set_only_x = utils.DatasetSplit(
            copy.deepcopy(val_dataset),
            idxs,
            runtime_poison=True,
            args=args,
            modify_label=False,
        )

    poisoned_val_only_x_loader = DataLoader(
        poisoned_val_set_only_x,
        batch_size=args.bs,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    # initialize a model, and the agents
    global_model = models.get_model(args.data, args).to(args.device).double()

    global_mask = {}
    neurotoxin_mask = {}
    updates_dict = {}
    n_model_params = len(
        utils.parameters_to_vector_sorted(global_model)  
    )
    params = {
        name: copy.deepcopy(global_model.state_dict()[name])
        for name in global_model.state_dict()
    }

    if args.aggr == "lockdown":
        sparsity = utils.calculate_sparsities(args, params, distribution=args.mask_init)
        mask = utils.init_masks(params, sparsity)
    # agent_secret_seed = random.sample(range(1, 101), args.num_agents)
    agent_secret_seed = [utils.initial_secret() for _ in range(args.num_agents)]
    args.lsh_dim = int(n_model_params * args.lsh_por)
    args.watermark_length = int(n_model_params * args.water_por)
    # args.lsh_size = 50
    # args.num_hash_tables = 10
    agents, agent_data_sizes = [], {}
    for _id in range(0, args.num_agents):
        if args.aggr == "lockdown":
            if args.same_mask == 0:
                agent = Agent_s(
                    _id,
                    args,
                    train_dataset,
                    user_groups[_id],
                    mask=utils.init_masks(params, sparsity),
                    backdoor_train_dataset=backdoor_train_dataset,
                    initial_alpha=agent_secret_seed[_id],
                )
            else:
                agent = Agent_s(
                    _id,
                    args,
                    train_dataset,
                    user_groups[_id],
                    mask=mask,
                    backdoor_train_dataset=backdoor_train_dataset,
                    initial_alpha=agent_secret_seed[_id],
                )
        else:
            agent = Agent(
                _id,
                args,
                train_dataset,
                user_groups[_id],
                backdoor_train_dataset=backdoor_train_dataset,
                secret_seed=agent_secret_seed[_id],
            )
        if args.byz:
            agent.is_malicious = 1 if _id < args.num_corrupt or _id >= args.num_agents-args.num_byz else 0
        else:
            agent.is_malicious = 1 if _id < args.num_corrupt else 0
        agent_data_sizes[_id] = agent.n_data
        agents.append(agent)

        logging.info(
            "build client:{} mal:{} data_num:{}".format(
                _id, agent.is_malicious, agent.n_data
            )
        )


    FPR_filters = {}
    aggregator = Aggregation(agent_data_sizes, n_model_params, args)

    criterion = nn.CrossEntropyLoss().to(args.device)
    agent_updates_dict = {}

    # num_spars = int(n_model_params * args.water_por)
    num_param = n_model_params
    idx = None
    mask = None
    if args.watermark:
        init_params = utils.parameters_to_vector_sorted(global_model)
        mask = torch.topk(torch.abs(copy.deepcopy(init_params)), k=args.watermark_length, largest=True)[1]
    alpha = args.alpha
    debug_alpha = {}
    alpha_scale = 1.0
    quanti_level = 1.0
    k = args.k
    best_acc = -1
    pre_exclude = None
    pre_rnd_global_updates = None
    pre_rnd_init_params = None
    curr_rnd_params = None
    # cur_rnd_k = 0
    # cur_rnd_delta = 1.0
    args.LSH_piece = True
    lsh = LSH(args)
    error_portion = torch.zeros(len(agent_updates_dict))
    benign_idx = range(args.num_agents)
    # client_hash = defaultdict(int)
    client_hash = {}

    for i in range(args.num_agents):
        client_hash[i] = torch.zeros(args.lsh_size*args.num_hash_tables*lsh.piece_length, dtype=torch.float64).to(args.device)
    client_masks = None
    for rnd in range(1, args.rounds + 1):
        logging.info("--------round {} ------------".format(rnd))
        rnd_global_params = utils.parameters_to_vector_sorted(global_model)
        # logging.info(f'Debug: Check the global params not change {torch.allclose(rnd_global_params,curr_rnd_params)}') if curr_rnd_params is not None else None
        curr_rnd_params = copy.deepcopy(rnd_global_params)
        if torch.isnan(curr_rnd_params).any().item():
            logging.info(f"Warning: Server update has nan in update, num of nan: {torch.isnan(curr_rnd_params).sum().item()}")
        timing_stats = {
            'client_embedding': [],
            'client_verification': [],
            'server_embedding':[],
            'server_verification': [],
            'server_lsh_filter': []
        }


        # args.lsh_size = 50
        # args.lsh_dim = num_spars
        # args.num_hash_tables = 10
        # args.multi_value = False
        
        message = None
        failed_clients = []
        rqim = None
        chaotic_alpha = None
        if args.watermark:
            # logging.info(f"Debug: Server generates a random message at beginning of each round, clients' global model no longer same")
            num_param = len(rnd_global_params)
            # num_spars = int(num_param * args.water_por)
            grads_unwater = copy.deepcopy(rnd_global_params)

        agent_updates_dict = {}
        chosen = np.random.choice(
            args.num_agents,
            math.floor(args.num_agents * args.agent_frac),
            replace=False,
        )
        chosen = sorted(chosen)
        if args.aggr == "lockdown":
            old_mask = [copy.deepcopy(agent.mask) for agent in agents]



        avg_embed_time = 0
        avg_detect_time = 0
        byz_params = []
        benign_params = []
        for agent_id in chosen:
            client_i = agents[agent_id]
            # client_msk = client_masks[agent_id] if client_masks is not None else mask
            client_msk = None
            client_alpha = None
            client_k = None
            test_parameters = None
            


              

            # client_alpha = alpha_scale if pre_rnd_global_updates is not None else agent_secret_seed[agent_id]
            # if agent_id in benign_idx and args.watermark:
            #     client_alpha = utils.cal_mzscore(update=client_i.update[pre_exclude],flat_global_model=pre_rnd_init_params[pre_exclude],global_update=pre_rnd_global_updates[pre_exclude],args=args) if pre_rnd_global_updates is not None else agent_secret_seed[agent_id]
            start_event.record()
            if args.watermark:
                print(f'Server -- Client {agent_id} has {agent_secret_seed[agent_id]}')
                i_round_seed = utils.generate_round_seed(agent_secret_seed[agent_id],rnd)                
                beta_id, delta,client_alpha,client_k, client_msk = utils.generat_params(i_round_seed,args.watermark_length,num_param,device=args.device)
                # beta_id, delta,client_alpha,client_k = beta_id.to(args.device), delta.to(args.device),client_alpha.to(args.device),client_k.to(args.device)
                # print(len(torch.unique(torch.tensor(mask)))==len(mask))
                rqim = QIM(delta=delta)
                num_param = len(rnd_global_params)
                

                # grads_unwater = copy.deepcopy(rnd_global_params)
                if args.authentication:
                    client_i_m = beta_id
                else:
                    client_i_m = rqim.random_msg(args.watermark_length)

                message = client_i_m
                test_parameters = {}
                # test_parameters['message'] = client_i_m
                # test_parameters['true parameter'] = rnd_global_params.clone()
                # test_parameters['hash'] = client_hash[agent_id]
                test_parameters['auth'] = beta_id
                test_parameters['delta'] = delta
                test_parameters['alpha'] = client_alpha
                test_parameters['k'] = client_k
                # if client_masks is not None:
                #     client_msk = client_masks[agent_id]

                test_parameters['mask'] = client_msk
                grads_water = utils.embedding_watermark_on_position(masks=client_msk,whole_grads=grads_unwater.clone(),Watermark=rqim,message=client_i_m,alpha=client_alpha,k=client_k)
                # cur_rnd_k = client_k

                # 
                grads_extract, mes = utils.detect_recover_on_position(masks=client_msk,whole_grads=copy.deepcopy(grads_water),Watermark=rqim,alpha=client_alpha,k=client_k)
                # logging.info(f"Smallest delta is {min(delta)}")
                # gene = torch.Generator().manual_seed(0)
                # for i in [0,1,3,5,7,9,10,15]:
                #     std = 1 / (10**i)
                #     noise = torch.clamp(torch.randn(num_param,generator=gene) *std,max=std)
                #     # print(max(noise))
                #     grads_noisesd = grads_water + noise.to(grads_water.device)
                #     incorrect_recover, incorrect_m = utils.detect_recover_on_position(masks=client_msk,whole_grads=copy.deepcopy(grads_noisesd),Watermark=rqim,alpha=client_alpha,k=client_k)
                #     wrong_m_num = torch.sum(incorrect_m==client_i_m)/args.watermark_length
                #     wrong_recover_mean = torch.sum(torch.abs(incorrect_recover-grads_unwater))/args.watermark_length
                #     logging.info(f'--- Round {rnd} inject noises {std}; recover mean difference: {wrong_recover_mean}, message ber: {wrong_m_num}')


                print("--------Test recovered params correct-------", torch.allclose(grads_extract, grads_unwater.clone()),torch.max(torch.abs(grads_extract-grads_unwater.clone())))
                print('-------Test extract message right------',torch.sum(mes==client_i_m))
                # mes1,dm_hat = utils.extract_on_position(masks=client_msk,whole_grads=copy.deepcopy(grads_water),Watermark=rqim,k=cur_rnd_k)
                # print(dm_hat.dtype)
                # print('-------Test extract message right (extract only) ------',torch.sum(mes1==client_i_m))
                # grads_extract1 = utils.recover_on_position(masks=client_msk,whole_grads=grads_water.clone(),Watermark=rqim,alpha=client_alpha,dm_hat=dm_hat)
                # print("--------Test recovered params correct-------", torch.allclose(grads_extract1, grads_unwater.clone()), torch.max(torch.abs(grads_extract1-grads_unwater.clone())))
                
                
                utils.vector_to_model(copy.deepcopy(grads_water), global_model)
                # grads_unwater_diff = torch.abs(grads_water[mask[0]:mask[1]] - copy.deepcopy(rnd_global_params)[mask[0]:mask[1]])


                # grads_unwater_diff = torch.abs(grads_water[client_msk] - copy.deepcopy(rnd_global_params)[client_msk])
                # exclude_mask = torch.ones_like(grads_water, dtype=torch.bool)
                # exclude_mask[client_msk] = False
                # logging.info(f"------Debug: BEFORE SENDING __evaluat the watermarked global model differences from unwatermarked model for client {agent_id}------")
                # logging.info(f"unwatermarked params same? {torch.allclose(grads_water[exclude_mask],copy.deepcopy(rnd_global_params)[exclude_mask])}, mean change on unwatermarked params: {torch.mean(torch.abs(grads_water[exclude_mask] - grads_unwater[exclude_mask])).item()}, {torch.mean(torch.abs(grads_water[exclude_mask] - grads_unwater[exclude_mask])).item()}")
                # logging.info(f"changed params: {grads_water.ne(copy.deepcopy(rnd_global_params)).sum().item()} out of {len(grads_water)}, expected {args.watermark_length}")
                # logging.info(f"Watermarked params Distorsion: mean {torch.mean(grads_unwater_diff).item()}, max {torch.max(grads_unwater_diff).item()}, min {torch.min(grads_unwater_diff).item()}")
                # logging.info(f"------Debug: SIGN CHECK --SHOULD BE SAME-----")
                # logging.info(f"Sign after embedding same for client {agent_id}? {torch.sum(torch.sign(grads_water[client_msk]) == torch.sign(rnd_global_params[client_msk])).item()} out of {num_spars} are same")
                # logging.info(f"Debugging: agent last round initial model params should be same as {torch.allclose(client_i.recovered_pre_rnd_init_params,pre_rnd_init_params)}") if pre_rnd_init_params is not None else None

            # if pre_rnd_init_params is not None:
            #     num_param = len(pre_rnd_init_params)
            #     num_spars = int(0.3 * num_param)
                
            #     min_params, _mask = torch.topk(torch.abs(copy.deepcopy(pre_rnd_init_params)), k=args.watermark_length, largest=True)
            #     true_hash = lsh.compute_lsh(pre_rnd_global_updates[_mask].unsqueeze(0))

            #     logging.info(f"Debug: Server embeds the watermark into model udpates, then sends to client {agent_id}, \nclient should use the previous rnd initial model params to calculate the watermarked updates first ")
            #     _rqim = QIM(delta=1)
                
            #     # print(f"Number of parameters: {args.watermark_length}")
            #     print(f"mask the Largest params")
            #     print(f'-----------FOR round {rnd-1}-----------')
            #     message = _rqim.random_msg(args.watermark_length)
            #     watered_host = utils.embedding_watermark_on_position(masks=_mask,whole_grads=copy.deepcopy(pre_rnd_init_params), Watermark=_rqim, message=message,alpha=1.0,k=0,quanti_factor=0.1)
            #     # print(f"num of changed params: {(watered_host!=pre_rnd_init_params).sum().item()} out of {len(watered_host)}")
            #     watered_hash = lsh.compute_lsh(watered_host[_mask].unsqueeze(0))
            #     print(torch.unique(true_hash), torch.unique(watered_hash))
            #     utils.compute_performance_drop(watered_host, global_model, criterion, val_loader, args, val_acc=best_acc)
            #     print(f"Matched hashes with watermarked model updates: ",utils.compare_hashes(true_hash, watered_hash),f"out of {args.lsh_size*args.num_hash_tables}")

            #     client_pre_updates = client_i.update
            #     client_i_hash = lsh.compute_lsh(client_pre_updates[_mask].unsqueeze(0))
            #     print(f"Matched hashes with client computed model updates: ",utils.compare_hashes(true_hash, client_i_hash),f"out of {args.lsh_size*args.num_hash_tables}")

            #     watered_host = copy.deepcopy(pre_rnd_init_params)
            #     watered_host[_mask] = torch.zeros(args.watermark_length, dtype=watered_host.dtype).to(args.device)
            #     print(f"Matched hashes with zeroed model updates: ",utils.compare_hashes(true_hash, lsh.compute_lsh(watered_host[_mask].unsqueeze(0))),f"out of {args.lsh_size*args.num_hash_tables}")
            #     # print(f"add zero, num of changed params: {(watered_host!=pre_rnd_init_params).sum().item()} out of {len(watered_host)}")
            #     # print(f"Unique values: {torch.unique(watered_host[_mask])}")
            #     utils.compute_performance_drop(watered_host, global_model, criterion, val_loader, args, val_acc=best_acc)

            # if pre_rnd_init_params is not None:
            #     logging.info(f"Debug: Server embeds the watermark into model udpates, then sends to client {agent_id}, \nclient should use the previous rnd initial model params to calculate the watermarked updates first ")
            #     _rqim = QIM(delta=1)
            #     num_param = len(pre_rnd_init_params)
            #     num_spars = int(0.4 * num_param)
            #     # print(f"Number of parameters: {num_spars}")
            #     print(f"use the largest updates for parameter selection")
            #     min_params, _mask = torch.topk(torch.abs(copy.deepcopy(pre_rnd_global_updates)), k=num_spars, largest=True)
            #     print(f"mask the Largest params")
            #     print(f'-----------FOR round {rnd-1}-----------')
            #     message = _rqim.random_msg(num_spars)
            #     watered_host = utils.embedding_watermark_on_position(masks=_mask,whole_grads=copy.deepcopy(pre_rnd_init_params), Watermark=_rqim, message=message,alpha=1.0,k=0,quanti_factor=0.1)
            #     print(f"num of changed params: {(watered_host!=pre_rnd_init_params).sum().item()} out of {len(watered_host)}")
            #     utils.compute_performance_drop(watered_host, global_model, criterion, val_loader, args, val_acc=best_acc)
            #     watered_host = copy.deepcopy(pre_rnd_init_params)
            #     watered_host[_mask] = torch.zeros(num_spars, dtype=watered_host.dtype).to(args.device)
            #     print(f"add zero, num of changed params: {(watered_host!=pre_rnd_init_params).sum().item()} out of {len(watered_host)}")
            #     print(f"Unique values: {torch.unique(watered_host[_mask])}")
            #     utils.compute_performance_drop(watered_host, global_model, criterion, val_loader, args, val_acc=best_acc)
            #     watered_host = copy.deepcopy(curr_rnd_params)
            #     print(f"current rnd params acc")
            #     utils.compute_performance_drop( watered_host, global_model, criterion, val_loader, args, val_acc=best_acc)
            #     watered_host[_mask] = pre_rnd_init_params[_mask]
            #     print(f"use previous initial params at watermarked positions, see performance")
            #     utils.compute_performance_drop(watered_host, global_model, criterion, val_loader, args, val_acc=best_acc)
            # if pre_rnd_global_updates is not None:
            #     logging.info(f"Debug: Server embeds the watermark into model udpates, then sends to client {agent_id}, \nclient should use the previous rnd initial model params to calculate the watermarked updates first ")
            #     _rqim = QIM(delta=1)
            #     num_param = len(pre_rnd_global_updates)
            #     num_spars = int(0.3 * num_param)
            #     print(f"Number of parameters: {num_spars}")
            #     cur_par = copy.deepcopy(pre_rnd_init_params)
            #     min_params, _mask = torch.topk(torch.abs(copy.deepcopy(pre_rnd_global_updates)), k=num_spars, largest=False)
            #     print(f"{len(min_params)}")
            #     message = _rqim.random_msg(num_spars)
            #     watered_host = utils.embedding_watermark_on_position(masks=_mask,whole_grads=copy.deepcopy(pre_rnd_global_updates), Watermark=_rqim, message=message,alpha=1.0,k=0,quanti_factor=0.1)
            #     print(f"num of changed params: {(watered_host!=pre_rnd_global_updates).sum().item()} out of {len(watered_host)}")
            #     utils.compute_performance_drop(watered_host+cur_par, global_model, criterion, val_loader, args, val_acc=best_acc)
            #     watered_host = copy.deepcopy(pre_rnd_global_updates)
            #     watered_host[_mask] = torch.zeros(num_spars, dtype=watered_host.dtype).to(args.device)
            #     print(f"add zero, num of changed params: {(watered_host!=pre_rnd_global_updates).sum().item()} out of {len(watered_host)}")
            #     print(f"Unique values: {torch.unique(watered_host[_mask])}")
            #     utils.compute_performance_drop(watered_host+cur_par, global_model, criterion, val_loader, args, val_acc=best_acc)
                
            end_event.record()
            torch.cuda.synchronize()
            avg_embed_time += start_event.elapsed_time(end_event)
            timing_stats["server_embedding"].append(start_event.elapsed_time(end_event))
            if client_i.is_malicious and args.super_power:
                continue
            global_model = global_model.to(args.device)

            if args.aggr == "lockdown":
                update = client_i.local_train(
                    global_model,
                    criterion,
                    rnd,
                    global_mask=global_mask,
                    neurotoxin_mask=neurotoxin_mask,
                    updates_dict=updates_dict,
                    masks=client_msk,
                    w_alpha=client_alpha, k=client_k,
                    lsh=lsh
                ) if args.watermark else client_i.local_train(
                    global_model, criterion, rnd, global_mask=global_mask,neurotoxin_mask=neurotoxin_mask, lsh=lsh
                )
            else:
                update = client_i.local_train(
                    global_model, criterion, rnd, neurotoxin_mask=neurotoxin_mask,masks=client_msk,delta=delta,alpha=client_alpha, k=client_k, test_params=test_parameters, lsh=lsh
                ) if args.watermark else client_i.local_train(
                    global_model, criterion, rnd, neurotoxin_mask=neurotoxin_mask, lsh=lsh
                )
            if update is None:
                continue
                # print(f"server: received embedded norm for client {agent_id}", update.norm().item())
            # if args.watermark:
            #     logging.info(f"---Debug: AFTER LOCAL TRAINING--- for client {agent_id}------")
            #     logging.info(f"Client {agent_id} final model params is correct (should be true if correctly recovered): {torch.allclose(client_i.curr_final_model_params-client_i.update, curr_rnd_params)}")
            #     logging.info(f"Client {agent_id} final model params is correct (should be true): {torch.allclose(client_i.curr_final_model_params-client_i.update, client_i.recovered_pre_rnd_init_params)}")
            #     # logging.info(f"Client {agent_id} final model params is correct (should be false unless no watermark): {torch.allclose(client_i.curr_final_model_params-client_i.update, grads_water)}")
            #     logging.info(f"Client {agent_id} recieved model params is same as distributed (should be true): {torch.allclose(grads_water, client_i.received_global_model_params)}")
            #     logging.info(f"Client {agent_id} -- Using alpha {client_i.alpha} same with calculated: {debug_alpha[agent_id]}, differences = {abs(client_i.alpha-debug_alpha[agent_id])}") if pre_rnd_global_updates is not None else None
            #     logging.info(f"Client {agent_id} pre global update same as expected {torch.allclose(client_i.pre_rnd_mk_global_updates[pre_exclude], pre_rnd_global_updates[pre_exclude])}") if hasattr(client_i, 'pre_rnd_mk_global_updates') and pre_exclude is not None else None

            #     logging.info(f"---Debug: SIGN CHECK -----")
            #     print(f"Debugging: agent model params should be same as (should be false?) {torch.allclose(client_i.recovered_pre_rnd_init_params,pre_rnd_init_params)}") if pre_rnd_init_params is not None else None
            #     print(f"Debugging: agent model params should be same as {torch.allclose(client_i.curr_final_model_params,pre_rnd_init_params)}") if pre_rnd_init_params is not None else None
            #     print(f"Debugging: agent model params should be same as {torch.allclose(client_i.pre_rnd_mk_global_updates,pre_rnd_init_params)}") if pre_rnd_init_params is not None else None
            #     logging.info(f'parameter (Should be all same): {torch.sum(torch.sign(grads_water) == torch.sign(client_i.pre_rnd_mk_global_updates+pre_rnd_init_params)).item()} sum of signs are same of client watermarked params (before recover) and true updates out of {len(update)}') if pre_rnd_init_params is not None else None
            #     logging.info(f'parameter (should be all same): {torch.sum(torch.sign(grads_water[client_msk]) == torch.sign((client_i.pre_rnd_mk_global_updates+pre_rnd_init_params)[client_msk])).item()} sum of signs are same of client watermarked params and true updates out of {num_spars}') if pre_rnd_init_params is not None else None

            #     logging.info(f'updates (doesn\'t need to be same, but outside watermark need same): {torch.sum(torch.sign(client_i.pre_rnd_mk_global_updates) == torch.sign(pre_rnd_global_updates)).item()} sum of signs are same of client watermarked updates (before recover) and true updates out of {len(update)}') if hasattr(client_i,'pre_rnd_mk_global_updates') and pre_rnd_global_updates is not None else None
            #     logging.info(f'updates (need same): {torch.sum(torch.sign(client_i.pre_rnd_mk_global_updates[pre_exclude]) == torch.sign(pre_rnd_global_updates[pre_exclude])).item()} sum of signs are same of client watermarked updates and true global updates, without watermarked part, {torch.sum(pre_exclude)}') if hasattr(client_i,'pre_rnd_mk_global_updates') and pre_rnd_global_updates is not None else None

            
            # check = parameters_to_vector(
            #         [
            #             copy.deepcopy(global_model.state_dict()[name])
            #             for name in global_model.state_dict()
            #         ]
            #     )
            # if mask is not None:
            #     print(f"after agent {agent_id} model parameter:{check[mask[0]:mask[0]+5]}")
            #     print(f"agent {agent_id} update:{update[mask[0]:mask[0]+5]}")
            # else:
            #     print(f"after agent {agent_id} model parameter:{check[:10]}")
            #     print(f"agent {agent_id} update:{update[:10]}")
            # if torch.isnan(update).any().item():
            #     logging.info(f"Client {agent_id} has nan in update, num of nan: {torch.isnan(update).sum().item()}")
            # if args.watermark:
            #     if hasattr(client_i,'recovered_pre_rnd_init_params'):
            #         recover_param_error = torch.abs(curr_rnd_params - client_i.recovered_pre_rnd_init_params)
            #         logging.info(f"Client {agent_id} received params recover error: {torch.mean(torch.abs(curr_rnd_params - client_i.recovered_pre_rnd_init_params)).item()}, \nmax error: {torch.max(torch.abs(curr_rnd_params - client_i.recovered_pre_rnd_init_params)).item()}, min error: {torch.min(torch.abs(curr_rnd_params - client_i.recovered_pre_rnd_init_params)).item()}")

            #         if not torch.allclose(curr_rnd_params,client_i.recovered_pre_rnd_init_params):
            #             failed_clients.append(agent_id)
            #             logging.info(f"Client {agent_id} received params recover error: {torch.mean(torch.abs(curr_rnd_params - client_i.recovered_pre_rnd_init_params)).item()}, \nmax error: {torch.max(torch.abs(curr_rnd_params - client_i.recovered_pre_rnd_init_params)).item()}, min error: {torch.min(torch.abs(curr_rnd_params - client_i.recovered_pre_rnd_init_params)).item()}")
                # recover_udpates, m = utils.detect_recover_on_position(masks=client_msk,whole_grads=copy.deepcopy(update),alpha=client_alpha,k=cur_rnd_k,Watermark=copy.deepcopy(args.rqim)) if args.watermark else (update,None)
                
            
            
            # logging.info(f"Server detects and recovers the watermark for Client {agent_id} using client alpha {client_alpha}, k {client_k}")
            

            # # TEST: add noise to watermarked positions to see if it affects the acc
            # c_update = copy.deepcopy(update)
            # # logging.info(f"adding random noise to same position update to test acc")
            
            # num_param = len(rnd_global_params)
            # num_spars = int(num_param * args.water_por)
            # logging.info(f"adding random noise to different position update to test acc")
            # if num_spars >= num_param:
            #     idx = 0
            # else:
            #     idx = torch.randint(0, (num_param - num_spars),size=(1,)).item()
            msk_se = 1
            start_event.record()
            if args.authentication:
                # logging.info(f"Smallest delta is {min(delta)}")
                # for i in [0,1,3,5,7,9,10,15]:
                #     std = 1 / (10**i)
                #     noise = torch.randn(num_param,generator=gene) *std
                #     # print(max(noise))
                #     grads_noisesd = update + noise.to(update.device)
                #     incorrect_recover, incorrect_m = utils.detect_recover_on_position(masks=client_msk,whole_grads=copy.deepcopy(grads_noisesd),Watermark=rqim,alpha=client_alpha,k=client_k)
                #     wrong_m_num = torch.sum(incorrect_m==client_i_m)/num_spars
                #     wrong_recover_mean = torch.sum(torch.abs(incorrect_recover-grads_unwater))/num_spars
                #     logging.info(f'--- Round {rnd} inject noises {std}; recover mean difference: {wrong_recover_mean}, message ber: {wrong_m_num}')

                update,_auth_id = utils.detect_recover_on_position(masks=client_msk,whole_grads=update,alpha=client_alpha,k=client_k,Watermark=rqim)
                if not torch.allclose(_auth_id, beta_id) and args.lsh_filter:
                    print(f'Server -- Client {agent_id} Tamper-detected in round {rnd}!')
                    agent_updates_dict[agent_id] = None
                    continue
                k_seed = utils.hash_to_seed(_auth_id)
                k_g = torch.Generator().manual_seed(k_seed)
                k_in = torch.rand(args.watermark_length,generator=k_g).to(args.device)
                client_k = k_in
                msk_se = k_seed
                print(f'Server -- Client {agent_id} Successfully Verified!')
            # msk_gen = torch.Generator(device=args.device).manual_seed(msk_se)
            # client_msk = torch.randperm(len(rnd_global_params), generator=msk_gen, device=args.device)[:args.watermark_length]
            agent_updates_dict[agent_id],extracted_message = utils.detect_recover_on_position(masks=client_msk,whole_grads=update,alpha=client_alpha,k=client_k,Watermark=rqim) if args.watermark else (update,None)
            end_event.record()
            torch.cuda.synchronize()
            # avg_detect_time += start_event.elapsed_time(end_event)
            timing_stats["server_verification"].append(start_event.elapsed_time(end_event))
            if args.watermark:
                timing_stats["client_embedding"].append(client_i.embed_time)
                timing_stats["client_verification"].append(client_i.recover_time)
                logging.info(f"Recovers the client {agent_id} updates mean error: {torch.mean(torch.abs(agent_updates_dict[agent_id] - client_i.update)).item()}, max error: {torch.max(torch.abs(agent_updates_dict[agent_id] - client_i.update)).item()}, min error: {torch.min(torch.abs(agent_updates_dict[agent_id] - client_i.update)).item()}")
                # logging.info(f"server extracted message is same with client message? {(extracted_message==client_i.m).all()}")

            if args.lsh_filter and args.watermark:
                # recover_udpates = copy.deepcopy(agent_updates_dict[agent_id])
                client_hash[agent_id] = extracted_message[:lsh.lsh_size*lsh.num_hash_tables*lsh.piece_length]
            #     print(f"server extracted hash is same with client hash? {(client_hash[agent_id]==client_i.hash).all()}") 
                # print(f"server extracted message is same with client message? {(extracted_message==client_i.m).all()}")
            #     logging.info(f"changed params: {recover_udpates.ne(copy.deepcopy(update)).sum().item()} out of {len(recover_udpates)}, expected {num_spars}")
            #     grads_unwater_diff = torch.abs(recover_udpates[client_msk] - copy.deepcopy(update)[client_msk])
            #     exclude_mask = torch.ones_like(recover_udpates, dtype=torch.bool)
            #     exclude_mask[client_msk] = False
            #     logging.info(f"unwatermarked params same? {torch.allclose(recover_udpates[exclude_mask],copy.deepcopy(update)[exclude_mask])}, mean change on unwatermarked params: {torch.mean(torch.abs(recover_udpates[exclude_mask]-copy.deepcopy(update)[exclude_mask])).item()}")
            #     logging.info(f"Watermarked params Distorsion: mean {torch.mean(grads_unwater_diff).item()}, max {torch.max(grads_unwater_diff).item()}, min {torch.min(grads_unwater_diff).item()}")

            #     if hasattr(client_i,'update'):
            #         print(f"STD of updates is: {torch.std(client_i.update).item()}")
            #         print(f"STD of recovery error is: {torch.std(recover_udpates - client_i.update).item()}")
            #         print(f"Client {agent_id} has recovery mean error: {torch.mean(torch.abs(recover_udpates - client_i.update)).item()}, max error: {torch.max(torch.abs(recover_udpates - client_i.update)).item()}, min error: {torch.min(torch.abs(recover_udpates - client_i.update)).item()}")
            #         if not torch.allclose(recover_udpates,client_i.update):
            #             logging.info(f"Failed to recover Client {agent_id}'s updates! mean error: {torch.mean(torch.abs(recover_udpates - client_i.update)).item()}, max error: {torch.max(torch.abs(recover_udpates - client_i.update)).item()}, min error: {torch.min(torch.abs(recover_udpates - client_i.update)).item()}")
            #         # recover_update_error = torch.abs(recover_udpates - client_i.update)
            #         # logging.info(f"Client {agent_id} updates recover error: {torch.mean(torch.abs(recover_udpates - client_i.update)).item()}, \nmax error: {torch.max(torch.abs(recover_udpates - client_i.update)).item()}, min error: {torch.min(torch.abs(recover_udpates - client_i.update)).item()}")
            #     if hasattr(client_i,'m'):
            #         if not client_i.m.eq(message).all().item():
            #             logging.info(f"Client {agent_id} failed recovers the message!")
                    # logging.info(f"Client {agent_id} message correct: {client_i.m.eq(message.detach().cpu()).all().item()}, message: {client_i.m[:10]}")
                # logging.info(f"Client {agent_id} updates norm after recover: {recover_udpates.norm().item()}, true norm {client_i.update.norm().item()}")
            # norm = update.norm().item()
            # logging.info(f"Client {agent_id} update norm: {norm}")
            # print(f"detected message on server for client {agent_id}: {m[:10]}, true message: {message.detach().cpu()[:10] if args.watermark else None}")
            # base = client_i.update.clone() if hasattr(client_i,'update') else None
                # print("server: recovered_norm", agent_updates_dict[agent_id].norm().item(),
                #     "reconstruction_err", (agent_updates_dict[agent_id] - base).norm().item())
                # # also check close:
                # print("close?", torch.allclose(agent_updates_dict[agent_id], base, atol=1e-6))

            # agent_updates_dict[agent_id],m = utils.detect_recover_on_position(masks=mask,whole_grads=update,alpha=alpha,k=client_k,Watermark=args.rqim) if args.watermark else (update,None)
            if client_i.is_malicious and agent_id > args.num_corrupt:
                byz_params.append(client_i.update)  # avoid recoving, because it should happen in malicous clients own side, they know the true local model
            else: benign_params.append(agent_updates_dict[agent_id]) # assume byz don't know benign clients' true updates
           

            utils.vector_to_model(copy.deepcopy(rnd_global_params), global_model)
            # if agent_id == 3:
            #     break # for debug purpose, only one client per round

        # logging.info(f"Byzantine agent group attack with {args.byz_attack} method, with benign client watermarked info")
        if args.byz:
            # for i in benign_params:
            #     benign_params[i] = 
            byz_params = Attack(byz_params, benign_params)
            indx = 0
            for agent_id in range(args.num_corrupt+1,args.num_agents-args.num_byz+1):
                if client_i.is_malicious:
                    byz_update = byz_params[indx]
                    ag = client_i
                    if args.watermark:
                        _user_param = byz_update.clone()
                        byz_update = utils.embedding_watermark_on_position(
                            masks=client_msk, whole_grads=_user_param, Watermark=ag.qim, message=ag.m, alpha=ag.alpha,k=ag.k
                        )
                    agent_updates_dict[agent_id] = byz_update
                    indx += 1
                    # logging.info(f"Byzantine agent {agent_id} update norm after attack: {byz_update.norm().item()}")
                    # logging.info(f"Byzantine agent {agent_id} successfully attack! {not torch.allclose(byz_update, ag.update)}")
            logging.info(f"Byzantine agent group attack with {args.byz_attack} method")
        # logging.info(f"Checking if the current global model is same as rnd_global_params: {}")
        # logging.info("checking if not update, would the non watermarked global model change")
        utils.vector_to_model(copy.deepcopy(curr_rnd_params), global_model)

        check = utils.parameters_to_vector_sorted(global_model)
        # if not torch.allclose(check,curr_rnd_params):
        #     logging.warning("Global params have changed during client updates!")
        # aggregate params obtained by agents and update the global params
        # if args.aggr == "flgmm":
        #     updates_dict = aggregator.aggregate_updates(
        #     global_model, agent_updates_dict, epoch=rnd, g0=pre_rnd_global_updates #, masks=mask,alpha=alpha, k=client_k
        #     )
        # else:
        #     updates_dict = aggregator.aggregate_updates(
        #         global_model, agent_updates_dict
        #     )
        # check = utils.parameters_to_vector_sorted(global_model)
        # print(f"Debug: Server: aggregated updates norm: {(check - curr_rnd_params).norm().item()}")
        


        
        # global_state_before = curr_rnd_params.clone()
        # print("server: global_before_norm", global_state_before.norm().item())
        # print("server: aggregated_delta_norm", pre_rnd_updates.norm().item())
        # print("server: new_global_norm", (global_state_before + pre_rnd_updates).norm().item())
        # logging.info(f"Client {agent_id} update norm: {pre_rnd_updates.norm().item()}")
        # print(f"current model parameter:{check[:10]}")
        # # print(f"updates:{updates_dict[:10]}")
        # print(f"")
        # inference in every args.snap rounds
        # sign_info = torch.sign(pre_rnd_updates)
        # curr_init_params = copy.deepcopy(curr_rnd_params)
        # curr_mask = copy.deepcopy(mask)
        next_mask = None
        # lsh_new = LSH(args)
        start_event.record()
        if args.lsh_filter:
            client_masks = {}
            hash_lists = []
            next_mask = []
            succss_verify = [aid for aid, update in agent_updates_dict.items() if update is not None]
            # calculate the LSH for each client's updates to detect the benign clients
            start = time.time()
            if args.watermark:
                # Batch collect pre-computed hashes
                hash_lists = [client_hash[aid] * 2 - 1 for aid in succss_verify]
                server_hashes = torch.stack(hash_lists, dim=0)
            else:
            # for agent_id in agent_updates_dict:
            #     if agent_updates_dict[agent_id] is not None:
            #         succss_verify.append(agent_id)
            #         if args.watermark:
            #             hash_lists.append(client_hash[agent_id]*2-1)
            #         else:
                # BATCH TOPK: Calculate all indices at once (Major speedup)
                # Shape: [Num_Active_Agents, Model_Dim]
                all_updates = torch.stack([agent_updates_dict[aid] for aid in succss_verify])
                
                # indices shape: [Num_Active_Agents, lsh_dim]
                all_tops_params, all_topk_indices = torch.topk(torch.abs(all_updates), k=lsh.lsh_dim, dim=1, largest=True)
                
                # 2. Sequential LSH calls (since function only accepts single input)
                # We iterate over the pre-calculated GPU indices
                for i in range(len(succss_verify)):
                    # Pass the i-th row of indices to the LSH function
                    single_hash = lsh.compute_lsh(all_tops_params[i])
                    hash_lists.append(single_hash.flatten() * 2 - 1)
                        # i_topk, _ = torch.topk(torch.abs(agent_updates_dict[agent_id]), k=lsh.lsh_dim, largest=True)
                        # hash_lists.append(lsh.compute_lsh(i_topk).flatten()*2-1)
                server_hashes = torch.stack(hash_lists, dim=0)
            for i in agent_updates_dict:
                _, msk_i = torch.topk(torch.abs(agent_updates_dict[agent_id]), k=args.watermark_length, largest=True)
                client_masks[agent_id] = msk_i
            #     # next_mask.update(list(msk_i))
            #     next_mask.extend(msk_i)
            # mask = torch.unique(torch.tensor(next_mask))
            # mask.sort()
            end = time.time()
            logging.info(f'lsh compute time:{end - start}')
            # print(f"Length of next mask: {len(mask)}, {len(next_mask)}")
            # print('Hash Length is ',len(hash_lists[0]),len(server_hashes[0]))
            # print('Watermarked Length is',num_spars)
            ########## use normalized updates for detection
            # server_hashes = torch.stack(server_hashes, dim=0)
            # major_server_hashes = torch.sign(torch.sum(torch.sign(server_hashes),dim=0))
            # _benign_normalized_lsh, mps_normalize = utils.hash_mz(major_server_hashes,server_hashes,args.lambda_s)
            # FPR_filters['Nomalized ordered'] = torch.sum(torch.tensor(_benign_normalized_lsh)<args.num_corrupt)/len(_benign_normalized_lsh)
            # utils.meature_metrics(_benign_normalized_lsh,args)
            # print(f"Use Normalized updates hash (ORDERED) for detect: {_benign_normalized_lsh, mps_normalize}")



            # server_hashes = torch.stack(original_ordered_hashes, dim=0)
            # major_server_hashes = torch.sign(torch.sum(torch.sign(server_hashes),dim=0))
            # __benign_normalized_lsh, mpss_normalized = utils.hash_mz(major_server_hashes,server_hashes,args.lambda_s)
            # print(f"Use Normalized updates hash (UNORDERED) for detect: {__benign_normalized_lsh, mpss_normalized}")
            # FPR_filters['Nomalized UNordered'] = torch.sum(torch.tensor(__benign_normalized_lsh)<args.num_corrupt)/len(__benign_normalized_lsh)
            # utils.meature_metrics(__benign_normalized_lsh,args)


            start = time.time()
            # server_hashes = torch.stack(hash_lists, dim=0)
            major_server_hashes = torch.sign(torch.sum(torch.sign(server_hashes),dim=0))
            __benign_unnormalized_lsh, mpss_normalized = utils.hash_mz(major_server_hashes,server_hashes,args.lambda_s)
            error_portion = mpss_normalized
            end = time.time()
            logging.info(f'lsh filter time:{end - start}')
            # print(__benign_unnormalized_lsh)
            # print(f"Use Unnormalized updates hash (Ordered) for detect: {__benign_unnormalized_lsh, mpss_normalized}")
            # FPR_filters['Unnormalized Ordered'] = torch.sum(torch.tensor(__benign_unnormalized_lsh)<args.num_corrupt)/len(__benign_unnormalized_lsh)
            # utils.meature_metrics(__benign_unnormalized_lsh,args)




            # hash_lists = torch.stack(hash_lists, dim=0)
            # # print(f"hash bit: {torch.unique(hash_lists)}")
            # major_hash = torch.sign(torch.sum(torch.sign(hash_lists), dim=0))

            ########## use previous round global updates as reference for detection

            # if pre_rnd_global_updates is not None:
            #     pre_msk = torch.topk(torch.abs(pre_rnd_global_updates), k=lsh.lsh_dim, largest=True)[1]
            #     pre_update_hash = torch.sign(lsh.compute_lsh(pre_rnd_global_updates[pre_msk]) *2 -1).flatten()
            #     benign_idx_LSH,mpsa_l = utils.hash_mz(pre_update_hash,hash_lists,args.lambda_s)
            #     print(f"Debug: LSH based benign clients (WITH Previous Updates) are {benign_idx_LSH}, mpsa list {mpsa_l}")
            #     FPR_filters['Unomalized previous'] = torch.sum(torch.tensor(benign_idx_LSH)<args.num_corrupt)/len(benign_idx_LSH)
            #     utils.meature_metrics(benign_idx_LSH,args)

            # tda_l/ist = []    


            # # ################## use unnormalized, major sign lsh
            # mpsa_list = []
            # for i in range(hash_lists.shape[0]):
            #     mpsa_list.append(torch.sum(major_hash != hash_lists[i]).item()/len(major_hash))
            # mpsa_std = np.std(mpsa_list)
            # mpsa_med = np.median(mpsa_list)
            # # normalized z-score, the smaller the better
            # mzscore_mpsa = []
            # for i in range(len(mpsa_list)):
            #     mzscore_mpsa.append(np.abs(mpsa_list[i] - mpsa_med) / mpsa_std)
            # print(mzscore_mpsa)
            # benign_idx_LSH = [int(i) for i in np.argwhere(np.array(mzscore_mpsa) < args.lambda_s)]
            # FPR_filters['Unomalized major'] = torch.sum(torch.tensor(benign_idx_LSH)<args.num_corrupt)/len(benign_idx_LSH)
            # utils.meature_metrics(benign_idx_LSH,args)
            # print(f"Debug: LSH based benign clients are {benign_idx_LSH}, mpsa std {mpsa_std}, median {mpsa_med}, with lambda_s {args.lambda_s}")
            # # logging.info(f"Debug: Server generates a random message at beginning of each round, clients' global model no longer same")
            # # num_param = len(rnd_global_params)
            # # num_spars = int(num_param * args.water_por)
            # print(f"Number of watermarked parameters: {num_spars}")
            # # next_mask = []
            # # logging.info(f"Debug: Server updated mask positions as end of each round: {curr_mask} to {next_mask}")
            

            # # if the benign clients can recover the model, we can skip excluding the watermarked positions
            # exclude_masks = torch.ones_like(curr_init_params, dtype=torch.bool)
            # exclude_masks[curr_mask] = False
            # # no_watermark_update = self.update[exclude_masks]
            # no_watermark_curr_init_params = curr_init_params

            # local_updates = []
            # for agent_id in agent_updates_dict:
            #     update = agent_updates_dict[agent_id]
            #     local_updates.append(update)
                # no_watermark_local_update = update[exclude_masks]
                # local_updates.append(no_watermark_local_update)
            # if curr_init_params is not None:
            #     euclid_benign = utils.hash_euclid(ref_update=curr_init_params,local_updates=local_updates)
            #     FPR_filters['euclid'] = torch.sum(torch.tensor(euclid_benign)<args.num_corrupt)/len(euclid_benign)
            #     print(f"Using euclidean distance with clustering, the results is: {euclid_benign}")
            #     utils.meature_metrics(euclid_benign,args)
            # logging.info(f"Aligning and the local updates with model params, and r-qim doesn change sign, for client, just use the watermarked params and they will get exact same results")
            # al, qu, benign_idx = utils.alignIns_alpha_quanti(local_updates=local_updates,flat_global_model=curr_init_params, args=args) # for debugging purpose, the model already updated using alignins
            # FPR_filters['alignIns'] = torch.sum(torch.tensor(benign_idx)<args.num_corrupt)/len(benign_idx)
            # utils.meature_metrics(benign_idx,args)


            benign_cln = {}
            final_benign = [succss_verify[i] for i in __benign_unnormalized_lsh]
            # Use Normalized Ordered -> _benign_normalized_lsh
            # Use Normalized Unordered -> __benign_normalized_lsh
            # Use Unnormalized Unordered -> benign_idx_LSH
            # Use Unnormalized Ordered -> __benign_unnormalized_lsh
            for i in final_benign:
                benign_cln[i] = agent_updates_dict[i]
            agent_updates_dict = benign_cln
            # print(f"Debug: AlignIns based benign clients are {benign_idx}")
            # print(f"LSH detected benign clients differences with alignins: {set(benign_idx_LSH).difference(set(benign_idx))}: LSH benign: {benign_idx_LSH}, AlignIns detected benign clients: {benign_idx}")
            # print(FPR_filters)
            # Use average aggregation
            if args.aggr == "flgmm":
                updates_dict = aggregator.aggregate_updates(
                global_model, agent_updates_dict, epoch=rnd, g0=pre_rnd_global_updates #, masks=mask,alpha=alpha, k=client_k
                )
            else:
                updates_dict = aggregator.aggregate_updates(
                    global_model, agent_updates_dict
                )
            check = utils.parameters_to_vector_sorted(global_model)
            end_event.record()
            torch.cuda.synchronize()
            # avg_detect_time += start_event.elapsed_time(end_event)
            timing_stats["server_lsh_filter"].append(start_event.elapsed_time(end_event))
            # print(f"Debug: Server: aggregated updates norm: {(check - curr_rnd_params).norm().item()}")
            
            # # al, qu, benign_idx = utils.alignIns_alpha_quanti(local_updates=local_updates,flat_global_model=pre_rnd_updates, args=args)
            # if hasattr(aggregator,'benign_idx'):
            #     logging.info(f"Debug: -- Check benign index same: {aggregator.benign_idx == benign_idx} --")
            #     logging.info(f"Debug: -- With watermark detected benign clients are {aggregator.benign_idx}, without water benign clients are {benign_idx} --")
            #     benign_idx = aggregator.benign_idx
            # logging.info(f"Debug: Detected benign clients are {benign_idx}, with before alpha {al} and quantization level {qu}")
            # al_max = al
            # al_min = 1e10
            # al_avg = 0
            # curr_global_updates = copy.deepcopy(check) - curr_rnd_params
            # for i in benign_idx:
            #     al_i = utils.cal_mzscore(update=local_updates[i], flat_global_model=curr_init_params,global_update=curr_global_updates, args=args)
            #     if al_i > al_max:
            #         al_max = al_i
            #     if al_i < al_min:
            #         al_min = al_i
            #     al_avg += al_i
            # al_avg = al_avg / len(benign_idx)
            # range_alpha = al_max - al_min
            # q_max = max(range_alpha, al_max / 1.5,2*al_min)
            # q_min = min(al_max/2, al_min *2)
            # # logging.info(f"Debug: AlignIns BENIGN Info -- Max mzscore: {al_max}, Min mzscore: {al_min}, Avg mzscore: {al_avg},interval: {al_max-al_min}")
            # # logging.info(f"Debug: AlignIns BENIGN Info -- Max Q: {q_max}, Min Q: {q_min}, mid Q: {(q_max+q_min)/2}, Quantization Level: {qu}")
            # debug_malicious = []
            # for i in chosen:
            #     al_i = utils.cal_mzscore(update=local_updates[i], flat_global_model=curr_init_params,global_update=curr_global_updates, args=args)
            #     debug_alpha[i] = al_i
            #     if i not in benign_idx:
            #         debug_malicious.append((al_i))
            # # print('Debug: Client calculated alphas:', debug_alpha)
            # # logging.info(f"Debug: AlignIns MALICIOUS Info -- Max mzscore: {max(debug_malicious)}, Min mzscore: {min(debug_malicious)}, mid mzscore: {(max(debug_malicious)+min(debug_malicious))/2}")
            # alpha_scale = al_avg
            # # logging.info(f"quantization level using max-min of alpha of benign clients")
            # quanti_level = (al_max-al_min)+1e-5
            # pre_exclude = exclude_masks
            logging.info(f"Successfully verified clients: {succss_verify}, total {len(succss_verify)}")
            logging.info(f"Updating benign clients {benign_cln.keys()}")
        else:
            # Use average aggregation
            if args.aggr == "flgmm":
                updates_dict = aggregator.aggregate_updates(
                global_model, agent_updates_dict, epoch=rnd, g0=pre_rnd_global_updates #, masks=mask,alpha=alpha, k=client_k
                )
            else:
                updates_dict = aggregator.aggregate_updates(
                    global_model, agent_updates_dict
                )
            check = utils.parameters_to_vector_sorted(global_model)
            # print(f"Debug: Server: aggregated updates norm: {(check - curr_rnd_params).norm().item()}")
        if args.watermark:
            logging.info(f"Server: Average embedding time: {sum(timing_stats['server_embedding'])/len(timing_stats['server_embedding'])}")
            logging.info(f"Server: Average extract and recover time: {sum(timing_stats['server_verification'])/len(timing_stats['server_verification'])}")
            logging.info(f"Server: LSH calculation time: {timing_stats["server_lsh_filter"]}")
            logging.info(f"Client: Average embedding time: {sum(timing_stats['client_embedding'])/len(timing_stats['client_embedding'])}")
            logging.info(f'Client: Average extraction and recover time: {sum(timing_stats['client_verification'])/len(timing_stats['client_verification'])}')
        pre_rnd_global_updates, pre_rnd_init_params, curr_rnd_params = (
                    check - curr_rnd_params, 
                    curr_rnd_params, 
                    check
                )
        logging.info("---------Test {} ------------".format(rnd))
        if rnd % args.snap == 0:
            if args.aggr != "lockdown":
                val_acc = utils.get_loss_n_accuracy(
                    global_model, criterion, val_loader, args, rnd, args.num_target
                )
                asr = utils.get_loss_n_accuracy(
                    global_model,
                    criterion,
                    poisoned_val_loader,
                    args,
                    rnd,
                    num_classes=args.num_target,
                )
                poison_acc = utils.get_loss_n_accuracy(
                    global_model,
                    criterion,
                    poisoned_val_only_x_loader,
                    args,
                    rnd,
                    args.num_target,
                )
            else:
                test_model = copy.deepcopy(global_model)

                # CF
                for name, param in test_model.named_parameters():
                    mask = 0
                    for id, agent in enumerate(agents):
                        mask += old_mask[id][name].to(args.device)
                    param.data = torch.where(
                        mask.to(args.device) >= args.theta_ld,
                        param,
                        torch.zeros_like(param),
                    )
                val_acc = utils.get_loss_n_accuracy(
                    test_model, criterion, val_loader, args, rnd, args.num_target
                )
                asr = utils.get_loss_n_accuracy(
                    test_model,
                    criterion,
                    poisoned_val_loader,
                    args,
                    rnd,
                    args.num_target,
                )
                poison_acc = utils.get_loss_n_accuracy(
                    test_model,
                    criterion,
                    poisoned_val_only_x_loader,
                    args,
                    rnd,
                    args.num_target,
                )
                del test_model

            logging.info("Clean ACC:              %.4f" % val_acc)
            logging.info("Attack Success Ratio:   %.4f" % asr)
            logging.info("Backdoor ACC:           %.4f" % poison_acc)

            if val_acc > best_acc:
                best_acc = val_acc
                best_asr = asr
                best_bcdr_acc = poison_acc
                logging.info("-----Best results So Far-----")
                # logging.info("Clean ACC:              %.4f" % best_acc)
                # logging.info("Attack Success Ratio:   %.4f" % best_asr)
                # logging.info("Backdoor ACC:           %.4f" % best_bcdr_acc)
                os.makedirs(f'/fred/oz410/project/FL/AlignIns_Changes/12-31_safe_Copy/AlignIns/outputs/{date}/', exist_ok=True)
                torch.save(global_model.state_dict(), f'/fred/oz410/project/FL/AlignIns_Changes/12-31_safe_Copy/AlignIns/outputs/{date}/{args.job}-best_model.pth')
                # utils.free_memory()
        logging.info("------------------------------".format(rnd))

    logging.info("Best results:")
    logging.info("Clean ACC:              %.4f" % best_acc)
    logging.info("Attack Success Ratio:   %.4f" % best_asr)
    logging.info("Backdoor ACC:           %.4f" % best_bcdr_acc)
    logging.info("Training has finished!")
