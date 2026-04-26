from datetime import date
date = date.today().strftime("%Y-%m-%d_m")
import torch
class QIM:
    def __init__(self, delta):
        # delta is the step size of quantization
        self.delta = delta
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.M_cls = 2.
    def embed(self, x:torch.Tensor, m:torch.Tensor,alpha=0.51,k=0, quanti_factor=None):
        """
        x is a vector of values to be quantized individually
        m is a binary vector of bits to be embeded
        returns: a quantized vector y
        """
        self.device = x.device
        scale = alpha
        d = self.delta
        dm = (m*d/self.M_cls).to(self.device)
        q_mk = quanti((x-dm-k), d) + (dm + k)
        y = q_mk * scale + x * (1 - scale)
        return y

    def detect(self, z,alpha=1,k=0,quanti_factor=None):
        """
        z is the received vector, potentially modified
        returns: a detected vector z_detected and a detected message m_detected
        """
        self.device = z.device
        d = self.delta
        shape = z.shape
        z = z.flatten()
        m_detected = torch.zeros_like(z, dtype=float)
        dm_hat = (quanti((z-k),d/self.M_cls)+k).to(self.device)
        z_hat = None
        if alpha is not None:
            scale = alpha
            z_hat = (z-scale * dm_hat)/ (1-scale)
        rough_m = torch.round((self.selective_round((dm_hat-k)/d)%1)*2)
        m_detected = rough_m
        m_detected = torch.reshape(m_detected,shape)
        return z_hat, m_detected.int().to(self.device)
   
    def selective_round(self,x:torch.tensor, threshold=0.99):
        frac = x % 1
        # Ensure the threshold is also a tensor for broadcasting
        threshold_tensor = torch.tensor(threshold, dtype=x.dtype, device=x.device)
        # The condition will be a boolean tensor
        condition = frac >= threshold_tensor
        # The true value and false value must be tensors of the same type and device
        true_val = torch.tensor(1.0, dtype=x.dtype, device=x.device)
        false_val = frac
        # Now, all arguments to torch.where are of compatible types and on the same device
        result_frac = torch.where(condition, true_val, false_val)
        return torch.floor(x) + result_frac
    
    def random_msg(self, l):
        """
        returns: a random binary sequence of length l
        """
        return torch.bernoulli(torch.full((l,), 0.5)).int().to(self.device)

def quanti(x, delta):
    """
    quantizes the input x with step size delta
    """
    if not isinstance(x,torch.Tensor):
        x = torch.tensor(x,dtype=torch.float64)
    return (torch.floor(x / delta) * delta)