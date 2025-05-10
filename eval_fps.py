import time
import torch
import torch.backends.cudnn as cudnn
from IRDPnet import IRDPNet
# from DABNet import DABNet

def compute_speed(model, input_size, device, iteration=100):
    torch.cuda.set_device(device)
    cudnn.benchmark = True

    model.eval()
    model = model.cuda()

    input = torch.randn(*input_size, device=device)

    for _ in range(50):
        model(input)

    print('=========Speed Testing=========')
    torch.cuda.synchronize()
    torch.cuda.synchronize()
    t_start = time.time()
    for _ in range(iteration):
        model(input)
    torch.cuda.synchronize()
    torch.cuda.synchronize()
    elapsed_time = time.time() - t_start

    speed_time = elapsed_time / iteration * 1000
    fps = iteration / elapsed_time

    print('Elapsed Time: [%.2f s / %d iter]' % (elapsed_time, iteration))
    print('Speed Time: %.2f ms / iter   FPS: %.2f' % (speed_time, fps))
    return speed_time, fps


if __name__ == '__main__':
    model =IRDPNet()
    size = "512,1024"  
    num_channels = 3  
    batch_size = 1  
    classes = 19  
    iter = 100   
    gpus = "0"  
    h, w = map(int,size.split(','))
    compute_speed(model, (batch_size, num_channels, h, w), int(gpus), iteration=iter)