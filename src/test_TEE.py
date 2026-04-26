
import gc
import numpy as np
import hashlib
import time
import argparse
def test_cpu_gpu_equivalence():
    np.random.seed(0)
    # Generate test data
    model_size = 6573120 # 6573120
    global_model = np.random.randn(model_size)
    
    
    n_model_params = len(
        global_model  
    )
    num_param = n_model_params
    watermark_length = int(n_model_params * 0.03)
    start_time = time.perf_counter()
    agent_secret_seed = initial_secret() 
    i_round_seed = generate_round_seed(agent_secret_seed,1)
    beta_id, delta,client_alpha,client_k, client_msk = generat_params(i_round_seed,watermark_length,num_param)
    
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    bench.record("Generation Operation", (execution_time) * 1000)
    rqim = QIM(delta=delta)
    authentication = True
    rnd_global_params = global_model
    del global_model
    # grads_unwater = rnd_global_params.clone()
    if authentication:
        client_i_m = beta_id
    else:
        client_i_m = rqim.random_msg(int(n_model_params * 0.03))
    start_time = time.perf_counter()
    _ = detect_recover_on_position(masks=client_msk,whole_grads=rnd_global_params,Watermark=rqim,alpha=client_alpha,k=client_k)
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    bench.record("Single extraction Operation", (execution_time) * 1000)

    start_time = time.perf_counter()
    grads_water = embedding_watermark_on_position(masks=client_msk,whole_grads=rnd_global_params,Watermark=rqim,message=client_i_m,alpha=client_alpha,k=client_k)
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    bench.record("Single embedding Operation", (execution_time) * 1000)
    del client_i_m
    gc.collect()
    # del rnd_global_params
    # gc.collect()

    start_time = time.perf_counter()
    _ = detect_recover_on_position(masks=client_msk,whole_grads=rnd_global_params,Watermark=rqim,alpha=client_alpha,k=client_k)
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    bench.record("Single extraction Operation", (execution_time) * 1000)

    parser = argparse.ArgumentParser(description="pass in a parameter")
    parser.add_argument("--lsh_filter", action="store_true", help="LSH defence, used alone for LSH only, with watermark & authentication for whole")
    parser.add_argument("--lsh_por", type=float, default=0.3, help="the portion of LSH tested")
    parser.add_argument("--lsh_size", type=int, default=100, help="the portion of LSH tested")
    parser.add_argument("--num_hash_tables", type=int, default=20, help="the portion of LSH tested")
    parser.add_argument("--water_por", type=float, default=0.03, help="the portion of watermarks to be embedded")
    args = parser.parse_args()
    args.lsh_dim = int(n_model_params * args.lsh_por)
    args.watermark_length = int(n_model_params * args.water_por)
    args.LSH_piece = True
    args.device = grads_water.device
    lsh = LSH(args)

    start_time = time.perf_counter()
    k = lsh.lsh_dim
    i_topk = np.partition(np.abs(grads_water), -k)[-k:]
    del grads_water
    gc.collect()
    hash = lsh.compute_lsh(i_topk)
    del i_topk
    m = np.zeros(watermark_length)
    end_time = time.perf_counter()
    lsh_time = end_time - start_time
    bench.record("Single LSH Operation", (lsh_time) * 1000)








class Benchmarker:
    def __init__(self):
        self.results = []

    def record(self, name, duration):
        self.results.append(f"{name} completed in {duration:.4f} ms")

    def report(self):
        print("\n--- FINAL BENCHMARK REPORT ---")
        for line in self.results:
            print(line)


bench = Benchmarker()

def embedding_watermark_on_position(masks,whole_grads,Watermark,message,alpha,k,quanti_factor=None,model=None):
    grad_unwater = whole_grads[masks]
    w_ = Watermark.embed(grad_unwater, m=message, alpha=alpha, k=k)
    whole_grads[masks] = w_
    return whole_grads

def initial_secret():
    import secrets
    # Generate a 256-bit seed (more standard for modern security)
    seed_256bit = secrets.token_hex(32) # 32 bytes = 256 bits
    return seed_256bit

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


class QIM:
    def __init__(self, delta):
        self.delta = delta
    def embed(self, x, m,alpha=0.51,k=0):
        """
        x is a vector of values to be quantized individually
        m is a binary vector of bits to be embeded
        returns: a quantized vector y
        """
        x = x.astype(float)
        scale = alpha
        d = self.delta
        dm = m*d/2.
        q_mk = quanti((x-dm-k), d) + (dm + k)
        y = q_mk * scale + x * (1 - scale)
        return y
    def detect(self, z, alpha=1.0, k=0, scale_delta=1):
        d = self.delta
        M_cls = 2.0
        shape = z.shape
        
        # 1. Flatten and cast in one step if necessary
        # If z is already float32/64, this is zero-copy
        z_flat = z.ravel().astype(np.float32, copy=False)
        
        # 2. Vectorized dm_hat calculation
        # quanti() must be vectorized; assuming it's a standard rounding/quantization
        dm_hat = quanti((z_flat - k), d / M_cls) + k
        
        # 3. Vectorized z_hat (The recovery step)
        # Pre-calculating the denominator is faster
        inv_scale = 1.0 / (1.0 - alpha)
        z_hat = (z_flat - alpha * dm_hat) * inv_scale
        
        # 4. Optimized Message Detection
        # Replacing the selective_round logic with a direct modulo/round approach
        # This replaces: np.round((self.selective_round((dm_hat-k)/d)%1)*2)
        m_detected = np.round(((dm_hat - k) / d % 1) * 2).astype(np.int32)
        
        return z_hat.reshape(shape), m_detected.reshape(shape)
    def selective_round(self,x, threshold=0.99):
        frac = x%1
        if np.allclose(frac,0.5):
            frac = 0.5
        if np.allclose(frac,0):
            frac = 0
        return np.floor(x) + np.where((x % 1) >= threshold, 1, (frac))
    def random_msg(self, l):
        """
        returns: a random binary sequence of length l
        """
        return np.random.choice((0, 1), l)

def quanti(x, delta):
    """
    quantizes the input x with step size delta
    """
    # the delta*floor[x/delta]
    # so floor will increase the distortion
    return np.floor(x / delta) * delta


def generat_params(seed, length, param_length):
    rng = np.random.default_rng(seed=seed)
    
    # 1. Faster integer generation for beta_id and masks
    # rng.integers is much faster than rng.choice for simple ranges
    beta_id = rng.integers(0, 2, size=length, dtype=np.int8)
    masks = rng.integers(0, param_length, size=length, dtype=np.int32)
    
    # 2. Bulk generate uniform randoms for delta and k_out in one go
    # Slicing a single large array is faster than two separate generator calls
    uniform_block = rng.random(size=length * 2, dtype=np.float32)
    delta = uniform_block[:length]
    k_out = uniform_block[length:]
    
    # 3. Use in-place operations for alpha to save memory allocation
    alpha = rng.standard_normal(length, dtype=np.float32)
    alpha *= 0.49
    alpha += 0.5
    np.clip(alpha, 0.501, 0.778, out=alpha) # In-place clip
    
    return beta_id, delta, alpha, k_out, masks

def detect_recover_on_position(masks, whole_grads, Watermark, alpha, k, quanti_factor=None, model=None):
    # Pass the slice directly. 
    # Note: If whole_grads is a large array, ensuring it's C-contiguous helps speed.
    grad_water = whole_grads[masks]
    
    # Perform detection
    recovered_segment, m_detected = Watermark.detect(grad_water, alpha=alpha, k=k)
    
    # Re-assign the recovered segment back to the original array
    whole_grads[masks] = recovered_segment
    
    return whole_grads, m_detected

import numpy as np

class LSH:
    __slots__ = ['_out_buf','lsh_size', 'lsh_dim', 'num_hash_tables',
                 'piece_length', 'd', 'n_i', 'sub_matrices', '_usable_len']

    def __init__(self, args):
        self.lsh_size      = args.lsh_size
        self.lsh_dim       = args.lsh_dim
        self.num_hash_tables = args.num_hash_tables
        self.piece_length  = (
            args.piece_length
            if hasattr(args, 'piece_length')
            else max(2, int(0.00001 * self.lsh_dim))
        )

        # Total hash bits: d = num_hash_tables * lsh_size
        self.d = self.num_hash_tables * self.lsh_size

        d_i       = self.d // self.piece_length   # hash bits per piece (local k)
        self.n_i  = self.lsh_dim // self.piece_length  # input dims per piece

        self._usable_len = self.n_i * self.piece_length

        rng = np.random.default_rng(seed=0)

        self.sub_matrices = rng.standard_normal(
            (self.piece_length, self.n_i, d_i)
        ).astype(np.float32)

    def compute_lsh(self, input_vector):
        # small initialiation + 88.0614 ms   in TEE 7935.9071 ms
        x = input_vector[:self._usable_len].reshape(self.piece_length, self.n_i)
        if x.dtype != np.float32:
            x = x.astype(np.float32)  
        projections = np.einsum('pi,pij->pj', x, self.sub_matrices)

        return projections.ravel() > 0
if __name__ == "__main__":
    test_cpu_gpu_equivalence()
    gc.collect()
    test_cpu_gpu_equivalence()
    bench.report()