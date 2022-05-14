import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # 0 GPU
os.system("taskset -p -c 0-3 %d" % os.getpid()) #0-1-2 CPU

from estimate_pose import *

np.random.seed(0)

if __name__=='__main__':
    if torch.cuda.is_available(): d, t = 'cuda', 'torch.cuda.FloatTensor'
    else: d, t = 'cpu', 'torch.FloatTensor'
    
    device = torch.device(d)
    torch.set_default_tensor_type(t)

    run(device)