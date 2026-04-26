import copy
import time

import torch
import utils
from torch.nn.utils import parameters_to_vector,vector_to_parameters
from torch.utils.data import DataLoader
from watermarks.modi_qim import QIM
from attacks import attack
from LshCls import LSH
class Agent():
    def __init__(self, id, args, train_dataset=None, data_idxs=None, mask=None, backdoor_train_dataset=None,secret_seed=1):
        self.id = id
        self.args = args
        self.error = 0
        self.logging = args.logging
        self.logging = args.logging
        self.hessian_metrix = []
        self.secret_seed = secret_seed
        
        # self.hash = torch.zeros(args.lsh_size*args.num_hash_tables, dtype=torch.float64).to(args.device)

        if self.id >= args.num_agents-args.num_byz:
            self.Attack  = attack(args.byz_attack)
        if args.watermark:
            self.rqim = QIM
            # self.alpha = args.alpha
            # self.k = args.k
        # get datasets, fedemnist is handled differently as it doesn't come with pytorch
        if self.args.data != "tinyimagenet":
            self.train_dataset = utils.DatasetSplit(train_dataset, data_idxs)

            # for backdoor attack, agent poisons his local dataset
            if self.id < args.num_corrupt and self.args.attack != 'non' and self.args.data != 'sen140':

                self.clean_backup_dataset = copy.deepcopy(train_dataset)
                self.data_idxs = data_idxs
                utils.poison_dataset(train_dataset, args, data_idxs, agent_idx=self.id) # args.poison frac, default to 0.5


            elif self.id < args.num_corrupt and self.args.attack != 'non' and self.args.data == 'sen140':
                self.clean_backup_dataset = copy.deepcopy(train_dataset)
                self.data_idxs = data_idxs
                benign_part = data_idxs[:int(len(data_idxs) * (1 - self.args.poison_frac))]
                malicious_part = data_idxs[int(len(data_idxs) * (1 - self.args.poison_frac)):]

                self.train_dataset = utils.DatasetSplit_new(train_dataset, backdoor_train_dataset, benign_part, malicious_part, data_idxs)
        else:
            self.train_dataset = utils.DatasetSplit(train_dataset, data_idxs, runtime_poison=True, args=args,
                                                        client_id=id)
        # get dataloader
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.args.bs, shuffle=True, \
                                       num_workers=args.num_workers, pin_memory=False, drop_last=True)
        # size of local dataset
        self.n_data = len(self.train_dataset)

    def check_poison_timing(self, round):
        if round > self.args.cease_poison:
            self.train_dataset = utils.DatasetSplit(self.clean_backup_dataset, self.data_idxs)
            self.train_loader = DataLoader(self.train_dataset, batch_size=self.args.bs, shuffle=True, \
                                           num_workers=self.args.num_workers, pin_memory=False, drop_last=True)

    def local_train(self, global_model, criterion, round=None, neurotoxin_mask=None,masks=None,delta=None,alpha=None,k=None, test_params=None,lsh=None):
        # print(len(self.train_dataset))
        """ Do a local training over the received global model, return the update """
        # start = time.time()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        cur_rnd_params = utils.parameters_to_vector_sorted(global_model).detach()
        # self.received_global_model_params = copy.deepcopy(cur_rnd_params)
        # aggr_updates = initial_global_model_params - self.previous_global_model_params if hasattr(self,'previous_global_model_params') else torch.zeros_like(initial_global_model_params)
        # start = time.time()
        start_event.record()
        if self.args.watermark:
            print(f'Client {self.id} has {self.secret_seed}')
            i_round_seed = utils.generate_round_seed(self.secret_seed,round)
            length = self.args.watermark_length
            
            self.beta_id, delta,self.alpha,self.k, masks = utils.generat_params(i_round_seed,length,len(cur_rnd_params),device=self.args.device)
            # self.beta_id, delta,self.alpha,self.k = self.beta_id.to(self.args.device), delta.to(self.args.device),self.alpha.to(self.args.device),self.k.to(self.args.device)
            # if hasattr(self,'update'):
            #     _, _masks = torch.topk(torch.abs(self.update), k=self.args.watermark_length, largest=True)
            #     print('mask',torch.allclose(_masks,test_params['mask']))
            #     masks = _masks.to(self.args.device)
            self.qim = self.rqim(delta=delta)
            ##############################################
            # print(f'agent {self.id} is using cosine similarity for alpha scaling')
            # self.alpha = self.initial_alpha*abs(torch.cosine_similarity(self.update,self.previous_global_model_params)) if hasattr(self, 'update') else self.initial_alpha

            # # here we assume a benign client will have true previous initial global parameters, and get the current watermarked global model parameters
            # self.pre_rnd_mk_global_updates = cur_rnd_params - self.recovered_pre_rnd_init_params if hasattr(self,'recovered_pre_rnd_init_params') else None

            # if test_params is not None:
            #     print(f'Client {self.id} has true previous initial global params provided for debugging')
            #     print('beta',torch.sum(self.beta_id-test_params['auth']))
            #     print('delta',torch.sum(delta-test_params['delta']))
            #     print('alpha',torch.sum(self.alpha-test_params['alpha']))
            #     print('k',torch.sum(self.k-test_params['k']))
            #     print('mask',torch.allclose(masks,test_params['mask']))
                # print(f"Client {self.id} previous initial global params match true params: {torch.allclose(no_watermark_curr_params, true_params[exclude_masks])}") if hasattr(self,'recovered_pre_rnd_init_params') else None
                # print(f"Client {self.id} previous initial global updates match true updates: {torch.allclose(no_watermark_curr_updates, true_params[exclude_masks]-no_watermark_previous_params)}") if hasattr(self,'recovered_pre_rnd_init_params') else None
            # #------------DEBUG-----------------
            # # self.alpha = alpha # use true alpha for Debugging
            # #------------DEBUG-----------------

            # self.k = k
            grad_water = copy.deepcopy(cur_rnd_params)
            if self.args.authentication:
                # g = torch.Generator().manual_seed(k)
                # chaotic_alpha = torch.randn(lsh.lsh_size*lsh.num_hash_tables,lsh.lsh_dim,generator=g).to(self.args.device).double()
                # chaotic_alpha = utils.chaotic_sequence(length=lsh.lsh_dim*lsh.lsh_size,init_value=torch.tensor(k)).to(self.args.device).view(lsh.lsh_dim,lsh.lsh_size)
                # self.alpha = 0.7
                # print(f"Client {self.id} -- Recovering with alpha {self.alpha}")
                # print(f"-------- Received model params for client {self.id}: {initial_global_model_params[masks[0]:masks[0]+5]} -------")
                recovered_params, received_id = utils.detect_recover_on_position(masks=masks,whole_grads=grad_water,alpha=self.alpha,k=self.k,Watermark=self.qim)
                # received_id, dm_hat = utils.detect_recover_on_position(masks,grad_water,self.qim,self.k)
                self.m = received_id
                # authentication verification
                print(torch.sum(received_id - self.beta_id), len(received_id))
                if not torch.allclose(received_id, self.beta_id):
                    print(f'Client {self.id} Tamper-detected in round {round}!')
                    return None
                print(f'Client {self.id} Successfully Verified!')
                # print(f"Client alpha is same as true alpha: {torch.allclose(self.alpha, alpha), torch.max(torch.abs(self.alpha-alpha))}") if alpha is not None else None
                # cur_rnd_params = utils.recover_on_position(masks,grad_water,self.qim,self.alpha,dm_hat=dm_hat)
                cur_rnd_params = recovered_params
                utils.vector_to_model_sorted(cur_rnd_params,global_model)
            else: 
                cur_rnd_params, self.m = utils.detect_recover_on_position(masks=masks,whole_grads=cur_rnd_params,alpha=self.alpha,k=self.k,Watermark=self.qim,model=global_model)
            # vector_to_parameters(initial_global_model_params,global_model.parameters())
            # self.recovered_pre_rnd_init_params = copy.deepcopy(utils.parameters_to_vector_sorted(global_model)).detach()
            # if test_params is not None:
            #     print(f'Client {self.id} has true previous initial global params provided for debugging')
            #     print(f"Client {self.id} curr initial global params match true params: {torch.allclose(self.recovered_pre_rnd_init_params, test_params['true parameter'])}") if hasattr(self,'recovered_pre_rnd_init_params') else None
            #     print(f"Client {self.id} curr initial global params match true params: {torch.max(torch.abs(self.recovered_pre_rnd_init_params-test_params['true parameter']))}")
            #     print(f"Client {self.id} extracted the correct message: {torch.allclose(self.m, test_params['message'])}")  # true
            #     print(f"Client {self.id} self hash is correctly extract: {torch.allclose(self.hash, test_params['hash'])}") if hasattr(self, 'hash') else None
            #     # print(f"Client {self.id} previous initial global updates match true updates: {torch.allclose(self.recovered_pre_rnd_init_params, true_params-no_watermark_previous_params)}") if hasattr(self,'recovered_pre_rnd_init_params') else None

            # self.pre_mask = copy.deepcopy(masks)
        # self.logging.info(torch.allclose(parameters_to_vector(global_model.parameters()),initial_global_model_params))
            # print(f"Recovered model params for client {self.id}: {parameters_to_vector(
            #     [global_model.state_dict()[name] for name in global_model.state_dict()]).detach()[masks[0]:masks[0]+5]}")
        # end = time.time()
        # recover_time = end - start
        # print(f'recover time time: {recover_time}')
        end_event.record()
        torch.cuda.synchronize()
        self.recover_time = start_event.elapsed_time(end_event)
        print(f'recover torch time: {self.recover_time}')

        if self.id < self.args.num_corrupt:
            self.check_poison_timing(round)
        global_model.train()
        optimizer = torch.optim.SGD(global_model.parameters(), lr=self.args.client_lr * (self.args.lr_decay) ** round,
                                    weight_decay=self.args.wd, momentum=self.args.momentum)

        regular_loss = 0.0
        for local_epoch in range(self.args.local_ep):
            start = time.time()
            old_gradient = {}
            old_gradient_mine = {}
            old_params = {}
            for i, (inputs, labels) in enumerate(self.train_loader):
                # if i == 0 and self.is_malicious:
                #     save_image(torch.cat([inputs[labels == self.args.target_class][:10]]), '%s_image.png' % self.id, normalize=True, nrow=10)
                optimizer.zero_grad()
                inputs, labels = inputs.to(device=self.args.device, non_blocking=True), \
                                 labels.to(device=self.args.device, non_blocking=True)
                outputs = global_model(inputs)
                # outputs = outputs[:, :]
                minibatch_loss = criterion(outputs, labels)
                # print(minibatch_loss)
                minibatch_loss.backward()
                if self.args.attack == "neurotoxin" and len(neurotoxin_mask) and self.id < self.args.num_corrupt:
                    for name, param in global_model.named_parameters():
                        param.grad.data = neurotoxin_mask[name].to(self.args.device) * param.grad.data
                if self.args.attack == "r_neurotoxin" and len(neurotoxin_mask) and self.id < self.args.num_corrupt:
                    for name, param in global_model.named_parameters():
                        param.grad.data = (torch.ones_like(neurotoxin_mask[name].to(self.args.device))-neurotoxin_mask[name].to(self.args.device) ) * param.grad.data
                optimizer.step()

                if self.args.attack == 'pgd' and self.id < self.args.num_corrupt and (i == len(self.train_loader) - 1):
                    if self.args.data == 'cifar10':
                        eps = torch.norm(cur_rnd_params) * 0.1
                    else:
                        eps = torch.norm(cur_rnd_params)

                    current_local_model_params = utils.parameters_to_vector_sorted(global_model).detach()
                    norm_diff = torch.norm(current_local_model_params - cur_rnd_params)
                    print('clip before: ', norm_diff)
                    if norm_diff > eps:
                        w_proj_vec = eps * (current_local_model_params - cur_rnd_params) / norm_diff + cur_rnd_params

                        print('clip after: ', torch.norm(w_proj_vec - cur_rnd_params))

                        new_state_dict = utils.vector_to_model_wo_load(w_proj_vec, global_model)    
                        global_model.load_state_dict(new_state_dict)

            end = time.time()
            train_time = end - start
            print("local epoch %d \t client: %d \t mal: %s \t loss: %.8f \t time: %.2f" % (local_epoch, self.id, str(self.is_malicious),
                                                                     minibatch_loss, train_time))

        with torch.no_grad():
            after_train = utils.parameters_to_vector_sorted(global_model).detach()
            self.update = after_train - cur_rnd_params
            # if hasattr(self, 'Attack'):
            #     print("individual attack for client", self.id)
            #     self.update = self.Attack([self.update],[])
            self.curr_final_model_params = after_train
            base = self.update.clone()
            print("client", self.id, "base_norm", base.norm().item(), "base_max", base.abs().max().item())

            print(f'agent {self.id} has mean {self.update.mean()}, and std {self.update.std()} after local training')
            # start = time.time()
            start_event.record()
            if self.args.watermark:
                # alpha = self.update.mean() if hasattr(self, 'update') else alpha
                # print(f"Client -- Embedding alpha is {alpha}")
                _user_param =  self.update.clone()
                start = time.time()
                if self.args.lsh_filter:
                    i_topk,_masks = torch.topk(torch.abs(self.update), k=lsh.lsh_dim, largest=True)
                    self.hash = lsh.compute_lsh(i_topk.unsqueeze(0)).flatten().int()
                    print(f"Client {self.id} LSH hash: {len(self.hash.flatten())}")
                    if not hasattr(self, 'm'):
                        self.m = torch.zeros(length)
                    self.m[:len(self.hash)] = self.hash.flatten()
                    end = time.time()
                    lsh_time = end - start
                    print(f'LSH time time: {lsh_time}')
                # print(f'Client detect watermark for client {self.id} at position {masks}, with alpha {self.alpha}, init alpha {self.initial_alpha}') #, alpha scale {distance})
                # print(f"quanti factor is {quanti_factor}")
                # msk_se = 1
                if self.args.authentication:
                    k_seed = utils.hash_to_seed(self.beta_id)
                    k_g = torch.Generator().manual_seed(k_seed)
                    k_in = torch.rand(length,generator=k_g).to(self.args.device)
                    msk_se = k_seed
                else:
                    k_in = self.k
                # msk_gen = torch.Generator(device=self.args.device).manual_seed(msk_se)
                # client_msk = torch.randperm(len(after_train), generator=msk_gen, device=self.args.device)[:self.args.watermark_length]
                update_param_w = utils.embedding_watermark_on_position(
                    masks=masks, whole_grads=_user_param, Watermark=self.qim, message=self.m, alpha=self.alpha,k=k_in
                )
                print('----- Inner Watermark embedded -----')
                # print(f"watermared model params for client {self.id}: {update_param_w[masks[0]:masks[0]+5]}")
                # print(f"unwatermared model params for client {self.id}: {self.update[masks[0]:masks[0]+5]}")
                # print("watermarked update mean: %.8f \t std: %.8f" % (update_param_w.mean(), update_param_w.std()))
                # print(f"------------ model updates:{self.update[masks[0]:masks[0]+5]} -------------\n")

                if self.args.authentication:
                    update_param_w  = utils.embedding_watermark_on_position(masks=masks,whole_grads=update_param_w,Watermark=self.qim,message=self.beta_id,alpha=self.alpha,k=self.k)
                    print('----- Outer Watermark embedded -----')
                # end = time.time()
                # embed_time = end - start
                # print(f'embed time time: {embed_time}')
                end_event.record()
                torch.cuda.synchronize()
                self.embed_time = start_event.elapsed_time(end_event)
                print(f'embed torch time: {self.embed_time}')
                return update_param_w.to(self.args.device)
            
            return self.update
