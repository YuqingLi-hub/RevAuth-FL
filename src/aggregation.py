import copy

import torch
from torch.nn.utils import parameters_to_vector
import numpy as np
import logging
from utils import vector_to_model, vector_to_name_param
from sklearn.cluster import KMeans, DBSCAN, MeanShift, estimate_bandwidth

import sklearn.metrics.pairwise as smp
from geom_median.torch import compute_geometric_median 


class Aggregation():
    def __init__(self, agent_data_sizes, n_params, args):
        self.agent_data_sizes = agent_data_sizes
        self.args = args
        self.server_lr = torch.tensor(args.server_lr, dtype=torch.float64, device=self.args.device)
        self.n_params = n_params
        if self.args.aggr == 'flgmm':
            self.r = []
            self.p = []
            self.f1 = []
            self.o = 0
            self.distances_matrix = []
            self.UCL = None
        if self.args.aggr == 'foolsgold':
            self.memory_dict = dict()
            self.wv_history = []
        
         
    def aggregate_updates(self, global_model, agent_updates_dict, epoch=0, g0=None,**kwargs):

        # print(f"type: {self.server_lr.dtype}, device: {self.n_params.dtype}")
        lr_vector = torch.Tensor([self.server_lr]*self.n_params).to(self.args.device, dtype=torch.float64)
        if self.args.aggr != "rlr":
            lr_vector = lr_vector
        else:
            lr_vector, _ = self.compute_robustLR(agent_updates_dict)
        # mask = torch.ones_like(agent_updates_dict[0])
        aggregated_updates = 0
        cur_global_params = parameters_to_vector(
            [global_model.state_dict()[name] for name in global_model.state_dict()]).detach()
        if self.args.aggr=='avg' or self.args.aggr == 'rlr' or self.args.aggr == 'lockdown':          
            aggregated_updates = self.agg_avg(agent_updates_dict)

        elif self.args.aggr == 'alignins':
            aggregated_updates = self.agg_alignins(agent_updates_dict, cur_global_params)
        elif self.args.aggr == 'mmetric':
            aggregated_updates = self.agg_mul_metric(agent_updates_dict, global_model, cur_global_params)
        elif self.args.aggr == 'foolsgold':
            aggregated_updates = self.agg_foolsgold(agent_updates_dict)
        elif self.args.aggr == 'signguard':
            aggregated_updates = self.agg_signguard(agent_updates_dict)
        elif self.args.aggr == "mkrum":
            aggregated_updates = self.agg_mkrum(agent_updates_dict)
        elif self.args.aggr == "rfa":
            aggregated_updates = self.agg_rfa(agent_updates_dict)
        elif self.args.aggr == "flgmm":
            aggregated_updates = self.agg_flgmm(agent_updates_dict,g0=g0, epoch=epoch, use_g0=self.args.use_g0,**kwargs)
        neurotoxin_mask = {}
        updates_dict = vector_to_name_param(aggregated_updates, copy.deepcopy(global_model.state_dict()))
        for name in updates_dict:
            updates = updates_dict[name].abs().view(-1)
            gradients_length = torch.numel(updates)
            _, indices = torch.topk(-1 * updates, int(gradients_length * self.args.dense_ratio))
            mask_flat = torch.zeros(gradients_length)
            mask_flat[indices.cpu()] = 1
            neurotoxin_mask[name] = (mask_flat.reshape(updates_dict[name].size()))

        cur_global_params = parameters_to_vector([ global_model.state_dict()[name] for name in global_model.state_dict()]).detach()
        # print((cur_global_params + lr_vector*aggregated_updates).dtype) already float64
        # print(f"lr_vector dtype: {lr_vector.dtype}, aggregated_updates dtype: {aggregated_updates.dtype}, cur_global_params dtype: {cur_global_params.dtype}")
        # lr_vector dtype: torch.float32, aggregated_updates dtype: torch.float64, cur_global_params dtype: torch.float64
        new_global_params =  (cur_global_params + lr_vector*aggregated_updates).double()
        vector_to_model(new_global_params, global_model)
        return updates_dict, neurotoxin_mask

    def agg_rfa(self, agent_updates_dict):
        local_updates = []

        for _id, update in agent_updates_dict.items():
            local_updates.append(update.cpu())

        n = len(local_updates)
        temp_updates = torch.stack(local_updates, dim=0).to(self.args.device)
        weights = torch.ones(n).to(self.args.device)
        gw = (compute_geometric_median(local_updates, weights.cpu()).median).to(self.args.device)
        for i in range(2):
            weights = torch.mul(weights, torch.exp(-1.0*torch.norm(temp_updates-gw, dim=1)))
            gw = (compute_geometric_median(local_updates, weights.cpu()).median).to(self.args.device)

        aggregated_model = gw
        return aggregated_model.to(self.args.device)

    def agg_alignins(self, agent_updates_dict, flat_global_model):
        local_updates = []
        benign_id = []
        malicious_id = []

        for _id, update in agent_updates_dict.items():
            local_updates.append(update)
            if self.args.byz and _id >= len(agent_updates_dict) - self.args.num_byz:
                malicious_id.append(_id)

            if _id < self.args.num_corrupt:
                malicious_id.append(_id)
            else:
                benign_id.append(_id)

        chosen_clients = malicious_id + benign_id
        num_chosen_clients = len(malicious_id + benign_id)
        inter_model_updates = torch.stack(local_updates, dim=0)

        tda_list = []
        mpsa_list = []
        major_sign = torch.sign(torch.sum(torch.sign(inter_model_updates), dim=0))
        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        for i in range(len(inter_model_updates)):
            _, init_indices = torch.topk(torch.abs(inter_model_updates[i]), int(len(inter_model_updates[i]) * self.args.sparsity))
            # calculate MPSA, the matching proportion of sign agreement, 0-1            
            mpsa_list.append((torch.sum(torch.sign(inter_model_updates[i][init_indices]) == major_sign[init_indices]) / torch.numel(inter_model_updates[i][init_indices])).item())

            # calculate TDA, the cosine similarity between each local update and (previous round) global model parameters? or update?
            tda_list.append(cos(inter_model_updates[i], flat_global_model).item())


        logging.info('TDA: %s' % [round(i, 4) for i in tda_list])
        logging.info('MPSA: %s' % [round(i, 4) for i in mpsa_list])


        ######## MZ-score calculation ########
        mpsa_std = np.std(mpsa_list)
        mpsa_med = np.median(mpsa_list)
        # normalized z-score, the smaller the better
        mzscore_mpsa = []
        for i in range(len(mpsa_list)):
            mzscore_mpsa.append(np.abs(mpsa_list[i] - mpsa_med) / mpsa_std)

        logging.info('MZ-score of MPSA: %s' % [round(i, 4) for i in mzscore_mpsa])
        
        tda_std = np.std(tda_list)
        tda_med = np.median(tda_list)
        mzscore_tda = []
        for i in range(len(tda_list)):
            mzscore_tda.append(np.abs(tda_list[i] - tda_med) / tda_std)

        logging.info('MZ-score of TDA: %s' % [round(i, 4) for i in mzscore_tda])

        ######## Anomaly detection with MZ score ########

        benign_idx1 = set([i for i in range(num_chosen_clients)])
        # filter with MPSA mzscore, lower than threshold is benign
        benign_idx1 = benign_idx1.intersection(set([int(i) for i in np.argwhere(np.array(mzscore_mpsa) < self.args.lambda_s)]))
        benign_idx2 = set([i for i in range(num_chosen_clients)])
        benign_idx2 = benign_idx2.intersection(set([int(i) for i in np.argwhere(np.array(mzscore_tda) < self.args.lambda_c)]))

        benign_set = benign_idx2.intersection(benign_idx1)
        
        benign_idx = list(benign_set)
        if len(benign_idx) == 0:
            return torch.zeros_like(local_updates[0])

        benign_updates = torch.stack([local_updates[i] for i in benign_idx], dim=0)

        ######## Post-filtering model clipping ########
        
        updates_norm = torch.norm(benign_updates, dim=1).reshape((-1, 1))
        norm_clip = updates_norm.median(dim=0)[0].item()
        benign_updates = torch.stack(local_updates, dim=0)
        updates_norm = torch.norm(benign_updates, dim=1).reshape((-1, 1))
        updates_norm_clipped = torch.clamp(updates_norm, 0, norm_clip, out=None)
        # del grad_norm
        
        benign_updates = (benign_updates/updates_norm)*updates_norm_clipped

        correct = 0
        for idx in benign_idx:
            if idx >= len(malicious_id):
                correct += 1

        TPR = correct / len(benign_id)

        if len(malicious_id) == 0:
            FPR = 0
        else:
            wrong = 0
            for idx in benign_idx:
                if idx < len(malicious_id):
                    wrong += 1
            FPR = wrong / len(malicious_id)

        logging.info('benign update index:   %s' % str(benign_id))
        logging.info('selected update index: %s' % str(benign_idx))

        logging.info('FPR:       %.4f'  % FPR)
        logging.info('TPR:       %.4f' % TPR)

        current_dict = {}
        for idx in benign_idx:
            current_dict[chosen_clients[idx]] = benign_updates[idx]

        aggregated_update = self.agg_avg(current_dict)
        self.benign_idx = benign_idx
        return aggregated_update

    def agg_avg(self, agent_updates_dict):
        """ classic fed avg """

        sm_updates, total_data = 0, 0
        for _id, update in agent_updates_dict.items():
            n_agent_data = self.agent_data_sizes[_id]
            sm_updates +=  n_agent_data * update
            total_data += n_agent_data
        return  sm_updates / total_data

    
    def agg_mkrum(self, agent_updates_dict):
        krum_param_m = 10
        def _compute_krum_score( vec_grad_list, byzantine_client_num):
            krum_scores = []
            num_client = len(vec_grad_list)
            for i in range(0, num_client):
                dists = []
                for j in range(0, num_client):
                    if i != j:
                        dists.append(
                            torch.norm(vec_grad_list[i]- vec_grad_list[j])
                            .item() ** 2
                        )
                dists.sort()  # ascending
                score = dists[0: num_client - byzantine_client_num - 2]
                krum_scores.append(sum(score))
            return krum_scores

        benign_id = []
        malicious_id = []

        for _id, update in agent_updates_dict.items():
            # local_updates.append(update)
            if _id < self.args.num_corrupt:
                malicious_id.append(_id)
            else:
                benign_id.append(_id)

        # Compute list of scores
        __nbworkers = len(agent_updates_dict)
        krum_scores = _compute_krum_score(agent_updates_dict, self.args.num_corrupt)
        score_index = torch.argsort(
            torch.Tensor(krum_scores)
        ).tolist()  # indices; ascending
        score_index = score_index[0: krum_param_m]

        print('%d clients are selected' % len(score_index))
        return_updates = [agent_updates_dict[i] for i in score_index]


        return sum(return_updates)/len(return_updates)

    def compute_robustLR(self, agent_updates_dict):

        agent_updates_sign = [torch.sign(update) for update in agent_updates_dict.values()]  
        sm_of_signs = torch.abs(sum(agent_updates_sign))
        mask=torch.zeros_like(sm_of_signs)
        mask[sm_of_signs < self.args.theta] = 0
        mask[sm_of_signs >= self.args.theta] = 1
        sm_of_signs[sm_of_signs < self.args.theta] = -self.server_lr
        sm_of_signs[sm_of_signs >= self.args.theta] = self.server_lr
        return sm_of_signs.to(self.args.device), mask

    def agg_mul_metric(self, agent_updates_dict, global_model, flat_global_model):
        local_updates = []
        benign_id = []
        malicious_id = []

        for _id, update in agent_updates_dict.items():
            local_updates.append(update)
            if _id < self.args.num_corrupt:
                malicious_id.append(_id)
            else:
                benign_id.append(_id)

        chosen_clients = malicious_id + benign_id
        num_chosen_clients = len(malicious_id + benign_id)

        vectorize_nets = [update.detach().cpu().numpy() for update in agent_updates_dict.values()]

        cos_dis = [0.0] * len(vectorize_nets)
        length_dis = [0.0] * len(vectorize_nets)
        manhattan_dis = [0.0] * len(vectorize_nets)
        for i, g_i in enumerate(vectorize_nets):
            for j in range(len(vectorize_nets)):
                if i != j:
                    g_j = vectorize_nets[j]

                    cosine_distance = float(
                        (1 - np.dot(g_i, g_j) / (np.linalg.norm(g_i) * np.linalg.norm(g_j))) ** 2)   #Compute the different value of cosine distance
                    manhattan_distance = float(np.linalg.norm(g_i - g_j, ord=1))    #Compute the different value of Manhattan distance
                    length_distance = np.abs(float(np.linalg.norm(g_i) - np.linalg.norm(g_j)))    #Compute the different value of Euclidean distance

                    cos_dis[i] += cosine_distance
                    length_dis[i] += length_distance
                    manhattan_dis[i] += manhattan_distance

        tri_distance = np.vstack([cos_dis, manhattan_dis, length_dis]).T

        cov_matrix = np.cov(tri_distance.T)
        inv_matrix = np.linalg.inv(cov_matrix)

        ma_distances = []
        for i, g_i in enumerate(vectorize_nets):
            t = tri_distance[i]
            ma_dis = np.dot(np.dot(t, inv_matrix), t.T)
            ma_distances.append(ma_dis)

        scores = ma_distances
        print(scores)

        p = 0.3
        p_num = p*len(scores)
        topk_ind = np.argpartition(scores, int(p_num))[:int(p_num)]   #sort

        print(topk_ind)
        current_dict = {}

        for idx in topk_ind:
            current_dict[chosen_clients[idx]] = agent_updates_dict[chosen_clients[idx]]

        update = self.agg_avg(current_dict)

        return update
   
    def agg_foolsgold(self, agent_updates_dict):
        def foolsgold(updates):
            """
            :param updates:
            :return: compute similatiry and return weightings
            """
            n_clients = updates.shape[0]
            cs = smp.cosine_similarity(updates) - np.eye(n_clients)

            maxcs = np.max(cs, axis=1)
            # pardoning
            for i in range(n_clients):
                for j in range(n_clients):
                    if i == j:
                        continue
                    if maxcs[i] < maxcs[j]:
                        cs[i][j] = cs[i][j] * maxcs[i] / maxcs[j]
            wv = 1 - (np.max(cs, axis=1))

            wv[wv > 1] = 1
            wv[wv < 0] = 0

            alpha = np.max(cs, axis=1)

            # Rescale so that max value is wv
            wv = wv / np.max(wv)
            wv[(wv == 1)] = .99

            # Logit function
            wv = (np.log(wv / (1 - wv)) + 0.5)
            wv[(np.isinf(wv) + wv > 1)] = 1
            wv[(wv < 0)] = 0

            # wv is the weight
            return wv, alpha

        local_updates = []
        benign_id = []
        malicious_id = []

        for _id, update in agent_updates_dict.items():
            local_updates.append(update)
            if _id < self.args.num_corrupt:
                malicious_id.append(_id)
            else:
                benign_id.append(_id)

        names = malicious_id + benign_id
        num_chosen_clients = len(malicious_id + benign_id)

        client_updates = [update.detach().cpu().numpy() for update in agent_updates_dict.values()]
        update_len = np.array(client_updates[0].shape).prod()
        # print("client_updates size", client_models[0].parameters())
        # update_len = len(client_updates)
        # if self.memory is None:
        #     self.memory = np.zeros((self.num_clients, update_len))
        if len(names) < len(client_updates):
            names = np.append([-1], names)  # put in adv

        num_clients = num_chosen_clients
        memory = np.zeros((num_clients, update_len))
        updates = np.zeros((num_clients, update_len))

        for i in range(len(client_updates)):
            # updates[i] = np.reshape(client_updates[i][-2].cpu().data.numpy(), (update_len))
            updates[i] = np.reshape(client_updates[i], (update_len))
            if names[i] in self.memory_dict.keys():
                self.memory_dict[names[i]] += updates[i]
            else:
                self.memory_dict[names[i]] = copy.deepcopy(updates[i])
            memory[i] = self.memory_dict[names[i]]
        # self.memory += updates
        use_memory = False

        if use_memory:
            wv, alpha = foolsgold(None)  # Use FG
        else:
            wv, alpha = foolsgold(updates)  # Use FG
        # logger.info(f'[foolsgold agg] wv: {wv}')
        self.wv_history.append(wv)

        print(len(client_updates), len(wv))


        weighted_updates = [update * wv[i] for update, i in zip(agent_updates_dict.values(), range(len(wv)))]

        aggregated_model = torch.mean(torch.stack(weighted_updates, dim=0), dim=0)

        print(aggregated_model.shape)

        return aggregated_model
    

    def agg_signguard(self, agent_updates_dict):
        f = self.args.num_corrupt
        benign_id = []
        malicious_id = []

        for _id, update in agent_updates_dict.items():
            # local_updates.append(update)
            if _id < self.args.num_corrupt:
                malicious_id.append(_id)
            else:
                benign_id.append(_id)

        gradients = [v for v in agent_updates_dict.values()]
        num_users = len(gradients)
        all_set = set([i for i in range(num_users)])
        iters = 1
        # stack all the gradients to one
        grads = torch.stack(gradients, dim=0)
        grads[torch.isnan(grads)] = 0 # remove nan

        # gradient norm-based clustering, calculate the l2 norm of each gradient
        grad_l2norm = torch.norm(grads, dim=1).cpu().numpy()
        norm_max = grad_l2norm.max()
        norm_med = np.median(grad_l2norm)
        # initialize benign index set, filter for first time
        benign_idx1 = all_set
        benign_idx1 = benign_idx1.intersection(set([int(i) for i in np.argwhere(grad_l2norm > 0.1*norm_med)]))
        benign_idx1 = benign_idx1.intersection(set([int(i) for i in np.argwhere(grad_l2norm < 3.0*norm_med)]))

        ## sign-gradient based clustering
        num_param = grads.shape[1]
        # select a small portion (0.1) of parameters to calculate the sign gradient
        num_spars = int(0.1 * num_param)
        # filter from the all users set
        benign_idx2 = all_set

        dbscan = 0
        meanshif = int(1-dbscan)

        for it in range(iters):
            # randomly select a portion of parameters
            idx = torch.randint(0, (num_param - num_spars),size=(1,)).item()
            gradss = grads[:, idx:(idx+num_spars)]
            # get the sign of the gradients, and sum the sign gradients
            sign_grads = torch.sign(gradss)
            sign_pos = (sign_grads.eq(1.0)).sum(dim=1, dtype=torch.float32)/(num_spars)
            sign_zero = (sign_grads.eq(0.0)).sum(dim=1, dtype=torch.float32)/(num_spars)
            sign_neg = (sign_grads.eq(-1.0)).sum(dim=1, dtype=torch.float32)/(num_spars)
            # calculate the normalized sign proportion of each clients
            pos_max = sign_pos.max()
            pos_feat = sign_pos / (pos_max + 1e-8)
            zero_max = sign_zero.max()
            zero_feat = sign_zero / (zero_max + 1e-8)
            neg_max = sign_neg.max()
            neg_feat = sign_neg / (neg_max + 1e-8)
            # print("pos_feat", pos_feat.shape, "zero_feat", zero_feat.shape, "neg_feat", neg_feat.shape)
            feat = [pos_feat, zero_feat, neg_feat]
            sign_feat = torch.stack(feat, dim=1).cpu().numpy()

            # in paper they use MeanShift to cluster
            if dbscan:
                clf_sign = DBSCAN(eps=0.05, min_samples=2).fit(sign_feat)
                labels = clf_sign.labels_
                n_cluster = len(set(labels)) - (1 if -1 in labels else 0)
                num_class = []
                for i in range(n_cluster):
                    num_class.append(np.sum(labels==i))
                benign_class = np.argmax(num_class)
                benign_idx2 = benign_idx2.intersection(set([int(i) for i in np.argwhere(labels==benign_class)]))
            else:
                uniq, counts = np.unique(sign_feat, axis=0, return_counts=True)
                
                bandwidth = estimate_bandwidth(sign_feat, quantile=0.5, n_samples=min(50, len(sign_feat)))
                if bandwidth <= 0:
                    print("Unique rows:", len(uniq), " / total:", len(sign_feat))
                    print("Max duplicate count:", counts.max())
                    print("Warning: estimated bandwidth <= 0, defaulting to std or 1.0")
                    bandwidth = np.std(sign_feat) or 1.0
                # print(time.time(), "Meanshift clustering")
                # bandwidth = estimate_bandwidth(sign_feat, quantile=0.5, n_samples=50)
                ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, cluster_all=False)
                ms.fit(sign_feat)
                labels = ms.labels_
                cluster_centers = ms.cluster_centers_
                labels_unique = np.unique(labels)
                # print("labels_unique", labels_unique)
                n_cluster = len(labels_unique) - (1 if -1 in labels_unique else 0)
                num_class = []
                for i in range(n_cluster):
                    num_class.append(np.sum(labels==i))
                benign_class = np.argmax(num_class)
                benign_idx2 = benign_idx2.intersection(set([int(i) for i in np.argwhere(labels==benign_class)]))
                # print(time.time(), "Meanshift clustering end")
        benign_idx = list(benign_idx2.intersection(benign_idx1))
        # print("benign_idx", benign_idx, "len", len(benign_idx))
        # calculate the misclassified attackers (because the byzantine idx is smaller than benign idx)
        byz_num = (np.array(benign_idx)<f).sum()
        # print("byz_num", byz_num)

        grad_norm = torch.norm(grads, dim=1).reshape((-1, 1))
        norm_clip = grad_norm.median(dim=0)[0].item()
        grad_norm_clipped = torch.clamp(grad_norm, 0, norm_clip, out=None)
        grads_clip = (grads/grad_norm)*grad_norm_clipped
        
        global_grad = grads_clip[benign_idx].mean(dim=0)
        correct = 0
        for idx in benign_idx:
            if idx >= len(malicious_id):
                correct += 1

        TPR = correct / len(benign_id)

        if len(malicious_id) == 0:
            FPR = 0
        else:
            wrong = 0
            for idx in benign_idx:
                if idx < len(malicious_id):
                    wrong += 1
            FPR = wrong / len(malicious_id)

        logging.info('SignGuard benign update index:   %s' % str(benign_id))
        logging.info('SignGuard selected update index: %s' % str(benign_idx))

        logging.info('SignGuard FPR:       %.4f'  % FPR)
        logging.info('SignGuard TPR:       %.4f' % TPR)

        return global_grad # this is the attack success rate
    
    def agg_flgmm(self, agent_updates_dict,g0=None, epoch=0, use_g0=False,ccepochs=50, masks=None,alpha=None, k=None):
        from matplotlib import pyplot as plt
        import utils
        import os
        import seaborn as sns
        f = self.args.num_corrupt
        num_users = len(agent_updates_dict)
        save_dir = f'outputs/FLGMM/{self.args.job}'
        os.makedirs(save_dir, exist_ok=True)
        if len(self.distances_matrix) != num_users:
            self.distances_matrix = [[] for _ in range(num_users)]
        distances_matrix_this_round = []
        normal_id = []
        excluded_clients = []
        # normal_clients_dis = []
        # normal_clients_dis_mean = []
        # normal_std = []
        normal_dis = [[] for _ in range(num_users)]
        FedAvg_0 = self.agg_avg
        if use_g0 and g0 is not None:
            w_glob = g0
        else:
            w_glob = FedAvg_0(agent_updates_dict)
        excluded = []
        gradients = [v for v in agent_updates_dict.values()]
        # noisy_this_round = []
        noisy_clients = [i for i in range(f)]
        # Calculate the euclidean distance between local and centroid weights
        for idx, w_local in enumerate(gradients):
            distance = utils.euclidean_distance(w_local, w_glob)
            distances_matrix_this_round.append(distance)

        distances = distances_matrix_this_round
        distances_array = np.array(distances).reshape(-1, 1)

        # Utilize GMM to find the largest cluster, return all the weights follows largest cluster
        largest_cluster_data, bounds, means, covariances, weights = utils.decompose_normal_distributions(distances_array)
        # print(len(largest_cluster_data))
        mean = np.mean(largest_cluster_data)
        std = np.std(largest_cluster_data)
        # print("Mean of largest cluster:", mean)
        # print("Std of largest cluster:", std)
        # use mean and std to normalize the largest cluster, and store them into distanc_matrix for SPC
        for idx in range(num_users):
            self.distances_matrix[idx].append((distances[idx]-mean)/std)
            # if during the initial rounds, directly use GMM results to select normal clients
            if distances_matrix_this_round[idx] in largest_cluster_data and epoch < ccepochs:
                normal_id.append(idx)

        # Plot GMM in round 10
        if epoch == 10:
            flat_distances3 = distances_matrix_this_round
            plt.figure(figsize=(10, 6))
            plt.hist(flat_distances3, bins=50, color='grey', edgecolor='black', density=True, alpha=0.6)
            x = np.linspace(min(distances_array), max(distances_array), 1000)
            pdf_1 = weights[0] * (1 / (np.sqrt(2 * np.pi * covariances[0]))) * np.exp(
                -0.5 * ((x - means[0]) ** 2) / covariances[0])
            pdf_2 = weights[1] * (1 / (np.sqrt(2 * np.pi * covariances[1]))) * np.exp(
                -0.5 * ((x - means[1]) ** 2) / covariances[1])
            pdf_1 = pdf_1.reshape(-1)
            pdf_2 = pdf_2.reshape(-1)
            plt.plot(x, pdf_1, color='red', linestyle='-.', label='GMM Component 1')
            plt.plot(x, pdf_2, color='green', linestyle='-.', label='GMM Component 2')
            plt.title('Initial Distance Distribution with GMM Components')
            plt.xlabel('Distance')
            plt.ylabel('Density')
            plt.legend()
            plt.savefig(os.path.join(save_dir, f'GMM_distance_distribution_with_GMM_10rounds.png'))
            upper_bound = bounds[1]
            lower_bound = bounds[0]
            # use upper bounds to filter out the clients in the largest cluster
            for idx, client_distances in enumerate(self.distances_matrix):
                normal_dis[idx] = [d for d in client_distances if d <= upper_bound]
            flat_distances1 = [distance for client_list in normal_dis for distance in client_list]
            plt.figure(figsize=(10, 6))
            plt.hist(flat_distances1, bins=40, color='grey', edgecolor='black', density=True)
            sns.kdeplot(flat_distances1, color='red')
            plt.title(f'Final Distance Distribution ')
            plt.xlabel('Distance')
            plt.ylabel('Density')
            plt.savefig(os.path.join(save_dir, f'Final_distance_distribution_10round.png'))
            plt.close()
        # when the initial rounds is over, use GMM results as reference
        if epoch == ccepochs:
            # normalid = []
            # f_distances_matrix = [[] for _ in range(num_users)]
            # for each distance in initial rounds (normalized)
            all_distances = [distance for client_distances in self.distances_matrix for distance in client_distances]
            distances_array = np.array(all_distances)

            # Utilize GMM again, find the largest cluster and its bounds
            largest_cluster_data, bounds, means, covariances, weights = utils.decompose_normal_distributions(distances_array)
            upper_bound = bounds[1]
            lower_bound = bounds[0]
            # use the upper bounds to filter out the clients in the largest cluster
            for idx, client_distances in enumerate(self.distances_matrix):
                normal_dis[idx] = [d for d in client_distances if d <= upper_bound ]
            flat_distances3 = [distance for client_list in self.distances_matrix for distance in client_list]
            plt.figure(figsize=(10, 6))
            plt.hist(flat_distances3, bins=200, color='grey', edgecolor='black', density=True, alpha=0.6)
            x = np.linspace(min(distances_array), max(distances_array), 1000)
            pdf_1 = weights[0] * (1 / (np.sqrt(2 * np.pi * covariances[0]))) * np.exp(
                -0.5 * ((x - means[0]) ** 2) / covariances[0])
            pdf_2 = weights[1] * (1 / (np.sqrt(2 * np.pi * covariances[1]))) * np.exp(
                -0.5 * ((x - means[1]) ** 2) / covariances[1])
            pdf_1 = pdf_1.reshape(-1)
            pdf_2 = pdf_2.reshape(-1)
            plt.plot(x, pdf_1, color='red', linestyle='-.', label='GMM Component 1')
            plt.plot(x, pdf_2, color='green', linestyle='-.', label='GMM Component 2')
            plt.title('Initial Distance Distribution with GMM Components')
            plt.xlabel('Distance')
            plt.ylabel('Density')
            plt.legend()
            plt.savefig(os.path.join(save_dir,f'GMM_distance_distribution_with_GMM.png'))

            flat_distances1 = [distance for client_list in normal_dis for distance in client_list]
            plt.figure(figsize=(10, 6))
            plt.hist(flat_distances1, bins=50, color='grey', edgecolor='black', density=True)
            sns.kdeplot(flat_distances1, color='red')
            plt.title(f'Final Distance Distribution ')
            plt.xlabel('Distance')
            plt.ylabel('Density')
            plt.savefig(os.path.join(save_dir, f'Final_distance_distribution_round.png'))
            plt.close()

            # Erase clients in the component with a lager mean, by using GMM again on largest cluster
            largest_cluster_data_2, bounds_2, means, covariances, weights = utils.decompose_normal_distributions(
                largest_cluster_data)
            upper_bound_2 = bounds[1]
            lower_bound_2 = bounds[0]
            for idx, client_distances in enumerate(self.distances_matrix):
                normal_dis[idx] = [d for d in client_distances if d <= upper_bound_2]
            # find the average distance of each client
            client_means = [np.mean(distances) for distances in self.distances_matrix]
            # determine the upper control limit (UCL) of the entire distance
            excluded_clients, UCL, LCL = utils.plot_control_chart(np.arange(len(client_means)), client_means, normal_dis, save_dir)
            self.UCL = UCL
            print('Upper Control Limit:', UCL)
            print("GMM detects:", excluded_clients)
            
            self.r.append(utils.calculate_accuracy(excluded_clients, noisy_clients)[0])
            self.p.append(utils.calculate_accuracy(excluded_clients, noisy_clients)[1])
            recall = utils.calculate_accuracy(excluded_clients, noisy_clients)[0]
            pre = utils.calculate_accuracy(excluded_clients, noisy_clients)[1]
            print("Initial recall:", self.r[0])
            print("Initial precision:", self.p[0])
            if recall + pre == 0:
                ff = 0.0
            else:
                ff = 2 * recall * pre / (recall + pre)
            self.f1.append(ff)
            
            print("Initial f1score:", self.f1[0])

        if epoch > ccepochs:
            # use UCL to determine the normal clients
            print('UCL:', self.UCL)
            for idx, client_distances in enumerate(self.distances_matrix):
                if self.UCL is not None:
                    if client_distances[-1] < self.UCL:
                        normal_id.append(idx)
                    else:
                        excluded.append(idx)
                else:
                    break
            excluded_clients = excluded
            print("Anomaly:", excluded)
            print("Normal clients:", normal_id)
            self.r.append(utils.calculate_accuracy(excluded_clients, noisy_clients)[0])
            self.p.append(utils.calculate_accuracy(excluded_clients, noisy_clients)[1])
            recall = utils.calculate_accuracy(excluded_clients, noisy_clients)[0]
            pre = utils.calculate_accuracy(excluded_clients, noisy_clients)[1]
            if recall + pre == 0:
                ff = 0.0
            else:
                ff = 2 * recall * pre / (recall + pre)
            self.f1.append(ff)
            # ff = 2 * recall * pre / (recall + pre)
            # self.f1.append(ff)
            print("Recall:", self.r[self.o])
            print("Precision:", self.p[self.o])
            print("f1score:", self.f1[self.o])
            self.o += 1

        # Update global model
        if epoch < ccepochs:
            gradients_used = [gradients[i] for i in range(len(gradients)) if i in normal_id]
            excluded_clients = [i for i in range(len(gradients)) if i not in normal_id]
            # print('numbers of participants:',len(gradients_used))

        else:
            gradients_used = [gradients[i] for i in range(len(gradients)) if i not in excluded_clients]
            normal_id = [i for i in range(len(gradients)) if i not in excluded_clients]
            # print('numbers of participants:', len(gradients_used))
        updates_dict = {}
        for i in range(len(gradients_used)):
            updates_dict[i] = gradients_used[i]
            # print(gradients_used[i].shape)
            # if self.args.watermark:
            #     updates_dict[i], m = utils.detect_recover_on_position(masks=masks,whole_grads=gradients_used[i],alpha=alpha,k=k,Watermark=self.args.rqim) if self.args.watermark else (gradients_used[i],None)
            #     print(f'client has updates norm {updates_dict[i].norm().item()}')
            # else:
                # updates_dict[i] = gradients_used[i]
        if len(gradients_used) > 0:
            w_glob = FedAvg_0(updates_dict)
        byz_num = (np.array(normal_id)<f).sum()
        return w_glob