import os
from IRDPnet import IRDPNet
import timeit
from Dataset_Builder import build_dataset_train
from torch.autograd import Variable
import torch.nn as nn
import time
import torch
from utils.utils import setup_seed, init_weight, netParams
from utils.metric import get_iou
from utils.loss import ProbOhemCrossEntropy2d
from utils.lr_scheduler import WarmupPolyLR
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
import numpy as np

GLOBAL_SEED = 1234

def val(classes,val_loader, model):
    # evaluation mode
    model.eval()
    total_batches = len(val_loader)

    data_list = []
    for i, (input, label, size, name) in enumerate(val_loader):
        with torch.no_grad():
            input_var = Variable(input).cuda()
        start_time = time.time()
        output = model(input_var)
        time_taken = time.time() - start_time
        print("[%d/%d]  time: %.2f" % (i + 1, total_batches, time_taken))
        output = output.cpu().data[0].numpy()
        gt = np.asarray(label[0].numpy(), dtype=np.uint8)
        output = output.transpose(1, 2, 0)
        output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)
        data_list.append([gt.flatten(), output.flatten()])

    meanIoU, per_class_iu = get_iou(data_list, classes)
    return meanIoU, per_class_iu



def train( train_loader, model, criterion, optimizer, epoch,max_epochs):
    """
    args:
       train_loader: loaded for training dataset
       model: model
       criterion: loss function
       optimizer: optimization algorithm, such as ADAM or SGD
       epoch: epoch number
    return: average loss, per class IoU, and mean IoU
    """
    model.train()
    epoch_loss = []

    total_batches = len(train_loader)
    print("=====> the number of iterations per epoch: ", total_batches)
    st = time.time()
    for iteration, batch in enumerate(train_loader, 0):
        per_iter = total_batches
        max_iter = max_epochs * per_iter
        cur_iter = epoch *per_iter + iteration
        scheduler = WarmupPolyLR(optimizer, T_max=max_iter, cur_iter=cur_iter, warmup_factor=1.0 / 3,
                                 warmup_iters=500, power=0.9)
        lr = optimizer.param_groups[0]['lr']

        start_time = time.time()
        images, labels, _, _ = batch
        images = Variable(images).cuda()
        labels = Variable(labels.long()).cuda()

        output = model(images)
        loss = criterion(output, labels)

        optimizer.zero_grad()  # Set gradients to zero
        loss.backward()
        optimizer.step()  # Update model parameters

        scheduler.step()  # Adjust learning rate AFTER optimizer step

        epoch_loss.append(loss.item())
        time_taken = time.time() - start_time

        print('=====> epoch[%d/%d] iter: (%d/%d) \tcur_lr: %.6f loss: %.3f time:%.2f' % (epoch + 1, max_epochs,
                                                                                         iteration + 1, total_batches,
                                                                                         lr, loss.item(), time_taken))

    time_taken_epoch = time.time() - st
    remain_time = time_taken_epoch * (max_epochs - 1 - epoch)
    m, s = divmod(remain_time, 60)
    h, m = divmod(m, 60)
    print("Remaining training time = %d hour %d minutes %d seconds" % (h, m, s))

    average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)

    return average_epoch_loss_train, lr







def train_model(train_type,max_epochs,input_size,random_mirror,random_scale,num_workers,lr,batch_size,resume,classes,cuda,gpus,num_train,num_val):
    h, w = map(int,input_size.split(','))
    input_size = (h, w)
    print("=====> input size:{}".format(input_size))


    if cuda:
        print("=====> use gpu id: '{}'".format(gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus
        if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

    # set the seed
    setup_seed(GLOBAL_SEED)
    print("=====> set Global Seed: ", GLOBAL_SEED)

    cudnn.enabled = True
    print("=====> building network")

    # build the model and initialization
    model =IRDPNet()
    init_weight(model, nn.init.kaiming_normal_,
                nn.BatchNorm2d, 1e-3, 0.1,
                mode='fan_in')

    print("=====> computing network parameters and FLOPs")
    total_paramters = netParams(model)
    print("the number of parameters: %d ==> %.2f M" % (total_paramters, (total_paramters / 1e6)))

    # load data and data augmentation
    datas, trainLoader, valLoader = build_dataset_train('cityscapes', input_size, batch_size, train_type,
                                                        random_scale, random_mirror, num_workers,num_train, num_val)

    print('=====> Dataset statistics')
    print("data['classWeights']: ", datas['classWeights'])
    print('mean and std: ', datas['mean'], datas['std'])

    # # define loss function, respectively
    # weight = torch.from_numpy(datas['classWeights'])

    min_kept = int(batch_size // len(gpus) * h * w // 16)
    criteria = ProbOhemCrossEntropy2d(use_weight=True, ignore_label=255,
                                          thresh=0.7, min_kept=min_kept)

    if cuda:
        criteria = criteria.cuda()
        if torch.cuda.device_count() > 1:
            print("torch.cuda.device_count()=", torch.cuda.device_count())
            gpu_nums = torch.cuda.device_count()
            model = nn.DataParallel(model).cuda()  # multi-card data parallel
        else:
            gpu_nums = 1
            print("single GPU for training")
            model = model.cuda()  # 1-card data parallel

    savedir = ('./checkpoint/' + 'cityscapes' + '/' + "IRDPnet" + 'bs'
                    + str(batch_size) + 'gpu' + str(gpu_nums) + "_" + str("train") + '/')

    if not os.path.exists(savedir):
        os.makedirs(savedir)

    start_epoch = 0

    # continue training
    if resume:
        if os.path.isfile(resume):
            checkpoint = torch.load(resume)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model'])
            # model.load_state_dict(convert_state_dict(checkpoint['model']))
            print("=====> loaded checkpoint '{}' (epoch {})".format(resume, checkpoint['epoch']))
        else:
            print("=====> no checkpoint found at '{}'".format(resume))

    model.train()
    cudnn.benchmark = True

    logFileLoc = savedir + "log.txt"
    if os.path.isfile(logFileLoc):
        logger = open(logFileLoc, 'a')
    else:
        logger = open(logFileLoc, 'w')
        logger.write("Parameters: %s Seed: %s" % (str(total_paramters), GLOBAL_SEED))
        logger.write("\n%s\t\t%s\t%s\t%s" % ('Epoch', 'Loss(Tr)', 'mIOU (val)', 'lr'))
    logger.flush()

    optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()), lr, momentum=0.9, weight_decay=1e-4)

    lossTr_list = []
    epoches = []
    mIOU_val_list = []

    print('=====> beginning training')
    for epoch in range(start_epoch, max_epochs):
        # training
        lossTr, lr = train(trainLoader, model, criteria, optimizer, epoch,max_epochs)
        lossTr_list.append(lossTr)

        # validation
        if epoch % 50 == 0 or epoch == (max_epochs - 1):
            epoches.append(epoch)
            mIOU_val, per_class_iu = val(classes, valLoader, model)
            mIOU_val_list.append(mIOU_val)
            # record train information
            logger.write("\n%d\t\t%.4f\t\t%.4f\t\t%.7f" % (epoch, lossTr, mIOU_val, lr))
            logger.flush()
            print("Epoch : " + str(epoch) + ' Details')
            print("Epoch No.: %d\tTrain Loss = %.4f\t mIOU(val) = %.4f\t lr= %.6f\n" % (epoch,
                                                                                        lossTr,
                                                                                        mIOU_val, lr))
        else:
            # record train information
            logger.write("\n%d\t\t%.4f\t\t\t\t%.7f" % (epoch, lossTr, lr))
            logger.flush()
            print("Epoch : " + str(epoch) + ' Details')
            print("Epoch No.: %d\tTrain Loss = %.4f\t lr= %.6f\n" % (epoch, lossTr, lr))

        # save the model
        model_file_name = savedir + '/model_' + str(epoch + 1) + '.pth'
        state = {"epoch": epoch + 1, "model": model.state_dict()}

        if epoch >= max_epochs - 10:
            torch.save(state, model_file_name)
        elif not epoch % 20:
            torch.save(state, model_file_name)

        # draw plots for visualization
        if epoch % 50 == 0 or epoch == (max_epochs - 1):
            # Plot the figures per 50 epochs
            fig1, ax1 = plt.subplots(figsize=(11, 8))

            ax1.plot(range(start_epoch, epoch + 1), lossTr_list)
            ax1.set_title("Average training loss vs epochs")
            ax1.set_xlabel("Epochs")
            ax1.set_ylabel("Current loss")

            plt.savefig(savedir + "loss_vs_epochs.png")

            plt.clf()

            fig2, ax2 = plt.subplots(figsize=(11, 8))

            ax2.plot(epoches, mIOU_val_list, label="Val IoU")
            ax2.set_title("Average IoU vs epochs")
            ax2.set_xlabel("Epochs")
            ax2.set_ylabel("Current IoU")
            plt.legend(loc='lower right')

            plt.savefig(savedir + "iou_vs_epochs.png")

            plt.close('all')

    logger.close()



if __name__ == '__main__':
    start = timeit.default_timer()
    train_type="train"  
    max_epochs= 50  
    input_size= "512,1024"
    random_mirror= True  
    random_scale= True   
    num_workers= 1
    lr=4.5e-2 
    batch_size= 2
    resume=" "  
    classes= 19  
    cuda= True   
    gpus= "0"    
    num_train= 50
    num_val=10
    #Training Model
    train_model(train_type,max_epochs,input_size,random_mirror,random_scale,num_workers,lr,batch_size,resume,classes,cuda,gpus,num_train,num_val)
    end = timeit.default_timer()
    hour = 1.0 * (end - start) / 3600
    minute = (hour - int(hour)) * 60