import torch
import torch.nn as nn
import utils
from resnet import ResNet9
import argparse
from torch.utils.data import DataLoader
import copy
from watermarks.modi_qim import QIM
class LSH():
    def __init__(self, args):
        self.args = args
        self.lsh_size = args.lsh_size
        self.lsh_dim = args.lsh_dim
        self.num_hash_tables = args.num_hash_tables
        self.multi_value = args.multi_value if hasattr(args, 'multi_value') else False
        self.input_piece =  args.LSH_piece if hasattr(args, 'LSH_piece') else False
        self.random_vectors = [torch.randn(self.lsh_dim, self.lsh_size).to(args.device).double() for _ in range(self.num_hash_tables)]
        self.piece_length = args.piece_length if hasattr(args, 'piece_length') else int(0.00001*self.lsh_dim)
        if self.input_piece:
            print('INPUT seperate into Pieces')
            self.random_vectors = [torch.randn(self.lsh_dim//self.piece_length, self.lsh_size).to(args.device).double() for _ in range(self.num_hash_tables)]

        self.quanti = args.LSH_quanti if hasattr(args, 'LSH_quanti') else 1
    def compute_lsh(self, input_vector):
        lsh_codes = []
        input_vector = input_vector.to(self.args.device)
        if self.input_piece:
            if len(input_vector.shape) <2:
                input_length = len(input_vector)
                input_vector = input_vector.unsqueeze(0)
            else:
                input_length = input_vector.shape[1]
            # print(input_length)
            pi_len = input_length // self.piece_length
            # print(pi_len)
            for i in range(self.piece_length):
                input = input_vector[:,pi_len*i : pi_len*(i+1)]
                # print(input.shape)
                for rv in self.random_vectors:
                    projections = (torch.matmul(input, rv))
                    lsh_code = (projections > 0).float()
                    lsh_codes.append(lsh_code)
        else:
            if not self.multi_value:
                for rv in self.random_vectors:

                    projections = (torch.matmul(input_vector, rv))
                    lsh_code = (projections > 0).float()
                    lsh_codes.append(lsh_code)
            else:
                for rv in self.random_vectors:
                    projections = (torch.matmul(input_vector, rv))
                    lsh_code = torch.zeros_like(projections).to(self.args.device)
                    lsh_code[projections > 0.5] = 2.0
                    lsh_code[(projections <= 0.5) & (projections >= -0.5)] = 1.0
                    lsh_code[projections < -0.5] = 0.0
                    lsh_codes.append(lsh_code)


        return torch.cat(lsh_codes, dim=0).to(self.args.device)
    