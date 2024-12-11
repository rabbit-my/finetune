import random
import time
import os
import torch.cuda
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms,datasets
import argparse
from network import backbone_dict,Projector,Classifier
from utils.log import AverageMeter,ProgressMeter
import wandb
import pandas as pd
import datetime
import numpy as np
from utils.grad_scaler import NativeScalerWithGradNormCount
import torch.backends.cudnn as cudnn

# def set_seed(seed):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     cudnn.deterministic = True
#     cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description="fine tune supervised classification")
    parser.add_argument('--data_dir',type=str,default='/home/codebase/Yinmi/CoOp/some_try4segoct/data_oct/train/',help = 'path to dataset')
    parser.add_argument('--epochs',type = int,default=100,help="number of epochs to train")
    parser.add_argument('--batch_size',type = int,default = 32,help = 'batch size for training')
    parser.add_argument('--lr',type = float,default= 1e-5,help='learning rate')
    parser.add_argument('--optimizer', default='AdamW', type=str, help='optimizer')
    parser.add_argument('--warmup_lr', type=float, default=1e-5,
                        help='warmup learning rate (default: 1e-5)')
    parser.add_argument('--min_lr', type=float, default=5e-7,
                        help='lower lr bound for cyclic schedulers that hit 0 (5e-7)')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--weight_decay', default=1e-2, type=float, help="weight_decay")
    parser.add_argument('--weight_decay_end', type=float, default=1e-3, help="""Final value of the
           weight decay. We use a cosine schedule for WD.""")
    # parser.add_argument('--momentum',type = float ,default= 0.9 ,help= 'SGD momentum' )
    parser.add_argument('--encoder_name',type = str,default= 'vit_b',help='backbone network')
    parser.add_argument('--hidden_size',type = int,default=512,help='projector hidden layer')
    parser.add_argument('--feature_dim',type = int,default= 128 ,help= 'projector feature dim' )
    parser.add_argument('--num_classes',type = int ,default= 3 ,help= 'number of classes')
    parser.add_argument('--checkpoint_dir',type=str,default='/home/codebase/Yinmi/CoOp/some_try4segoct/checkpoints_small/',help = 'path to checkpoint')
    # parser.add_argument( '--schedule',default=[60, 80],nargs="*",type=int,help="learning rate schedule (when to drop lr by a ratio)")
    parser.add_argument('--val1_dir', type=str, default='/home/codebase/Yinmi/CoOp/some_try4segoct/data_oct/test/', help='path to validation set 1')
    parser.add_argument('--val2_dir', type=str, default='/home/codebase/Yinmi/CoOp/some_try4segoct/data_oct/xiangya/', help='path to validation set 2')
    parser.add_argument('--csv_dir', type=str, default='/home/codebase/Yinmi/CoOp/some_try4segoct/result_csv_small/',help = 'path to val result')
    parser.add_argument('--print_freq',type = int,default=10,help = 'The frequency of printing in a batch')


    args = parser.parse_args()
    args.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    return args


def train_transform():
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(size=(256, 512), scale=(0.8, 1.0), ratio=(1.8, 2.2), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2)], p=0.5),
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.3124, 0.3124, 0.3124], std=[0.2206, 0.2206, 0.2206])
    ])
    return train_transforms
def val_transform():
    val_transforms = transforms.Compose([
        transforms.Resize(size=(256, 512), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.3124, 0.3124, 0.3124], std=[0.2206, 0.2206, 0.2206])
    ])
    return val_transforms

def make_train_loader(data_dir,batch_size):

    transform = train_transform()
    dataset = datasets.ImageFolder(root=data_dir,transform = transform)
    print(dataset)
    loader = DataLoader(dataset,batch_size=batch_size,shuffle=True,num_workers=16,pin_memory=True,drop_last=True)
    return loader

def make_val_loader(data_dir,batch_size):
    transform = val_transform()
    dataset = datasets.ImageFolder(root=data_dir,transform = transform)
    print(dataset)
    loader = DataLoader(dataset,batch_size=batch_size,shuffle=False,num_workers=16,pin_memory=True)
    return loader

"""create model"""

class Model_wrapper(nn.Module):
    def __init__(self, encoder, projector, classifier):
        super(Model_wrapper, self).__init__()
        self.encoder = encoder
        self.projector = projector
        self.classifier = classifier

    def forward(self, x):
        features = self.encoder(x)
        projected_features = self.projector(features)
        logits = self.classifier(projected_features)
        return logits, projected_features

def create_model(encoder_name, hidden_size, feature_dim, num_classes):
    encoder, dim_in = backbone_dict[encoder_name]
    projector = Projector(dim_in, hidden_size, feature_dim)
    classifier = Classifier(feature_dim, num_classes)
    model = Model_wrapper(encoder, projector, classifier)
    return model


def train(model, train_loader, val1_loader, val2_loader, criterion,optimizer, args):
    device = args.device
    niter_per_ep = len(train_loader)
    scaler = NativeScalerWithGradNormCount(optimizer=optimizer,amp=True,clip_grad=1.0)
    lr_schedule,wd_schedule = cosine_scheduler(args.lr,args.min_lr,args.weight_decay,args.weight_decay_end,args.epochs,niter_per_ep,args.warmup_epochs,args.warmup_lr)

    for epoch in range(args.epochs):
        batch_time = AverageMeter("Time", ":6.3f")
        data_time = AverageMeter("Data", ":6.3f")
        losses = AverageMeter("Loss", ":.4f")
        top1 = AverageMeter("Acc@1", ":6.2f")


        progress = ProgressMeter(
            len(train_loader),
            [batch_time, data_time, losses, top1],
            prefix="Epoch: [{}]".format(epoch),
        )
        model.train()
        end = time.time()
        for i, (images, labels) in enumerate(train_loader):
            data_time.update(time.time() - end)

            images, labels = images.to(device), labels.to(device)

            unique_labels, counts = torch.unique(labels.cpu(), return_counts=True)
            label_distribution = dict(zip(unique_labels.numpy(), counts.numpy()))
            print(f"Batch {i}: Label Distribution: {label_distribution}")

            with torch.cuda.amp.autocast():
                logits, features = model(images)
                loss = criterion(logits, labels)


            acc1 = accuracy(logits, labels, topk=(1,))
            losses.update(loss.item(), images.size(0))


            top1.update(acc1, images.size(0))

            scaler(loss,optimizer=optimizer,parameters=model.parameters(),update_grad=True)

            current_iter = epoch * niter_per_ep + i
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_schedule[current_iter]
                param_group['weight_decay'] = wd_schedule[current_iter]


            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0 and i != 0:
                progress.display(i)

        wandb.log({"acc":top1.avg, "loss":losses.avg})

        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            save_model(model, epoch, args.checkpoint_dir,optimizer,args)

        # print(f'Epoch {epoch}/{args.epochs - 1}, Loss: {losses.avg:.4f}, Acc@1: {top1.avg:.2f}')
        print(f'Epoch {epoch}/{args.epochs - 1}, Loss: {losses.avg:.4f},  Acc@1: {top1.avg:.2f}')
        val1_acc = validate(model, val1_loader, device, 'data_A_test', args.csv_dir,epoch)
        val2_acc = validate(model, val2_loader, device, 'xiangya', args.csv_dir,epoch)
        wandb.log({"data_A_test": val1_acc, "xiangya_val_acc": val2_acc})
    print("Finish training...")



def validate(model, val_loader, device, data_name, csv_dir,epoch):

    model.eval()
    predictions = []
    labels = []

    with torch.no_grad():
        for images, true_labels in val_loader:
            images, true_labels = images.to(device), true_labels.to(device)
            outputs,_ = model(images)
            _, predicted_labels = torch.max(outputs, 1)

            predictions.extend(predicted_labels.cpu().numpy())
            labels.extend(true_labels.cpu().numpy())

    accuracy = sum(p == l for p, l in zip(predictions, labels)) / len(predictions)

    df = pd.DataFrame({
        'Predicted Labels': predictions,
        'True Labels': labels
    })

    csv_path = os.path.join(csv_dir, f'{data_name}_validation_results_{epoch}_epoch.csv')
    df.to_csv(csv_path, index=False)
    return accuracy


def save_model(model, epoch, checkpoint_dir,optimizer,args):
    os.makedirs(checkpoint_dir,exist_ok=True)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_name = args.encoder_name
    model_path = os.path.join(checkpoint_dir, f'{model_name}_epoch{epoch}_{timestamp}.pth')
    torch.save({
        'epoch':epoch,
        'state_dict':model.state_dict(),
        'optimizer':optimizer.state_dict(),
        'arg':args
    },model_path)
    print(f"Model weights for {model_name} at epoch {epoch} saved to '{model_path}'")


def cosine_scheduler(base_lr, final_lr, base_wd, final_wd, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_iters = warmup_epochs * niter_per_ep

    warmup_lr = np.linspace(start_warmup_value, base_lr, warmup_iters) if warmup_epochs > 0 else np.array([])
    warmup_wd = np.linspace(final_wd, base_wd, warmup_iters)  # Start from final_wd to base_wd during warmup

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    cosined_lr = final_lr + 0.5 * (base_lr - final_lr) * (1 + np.cos(np.pi * iters / len(iters)))
    cosined_wd = final_wd + 0.5 * (base_wd - final_wd) * (1 + np.cos(np.pi * iters / len(iters)))
    lr_schedule = np.concatenate((warmup_lr, cosined_lr))
    wd_schedule = np.concatenate((warmup_wd, cosined_wd))
    return lr_schedule, wd_schedule

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        correct_1 = correct[:1].view(-1).float().sum(0, keepdim=True)
        acc1 = correct_1.mul_(100.0 / batch_size)
        return acc1.item()


def init_wandb(args):

    current_time = datetime.datetime.now()
    date_str = current_time.strftime("%Y-%m-%d")
    hour_str = current_time.strftime("%H")

    run_name = f"{args.encoder_name}_epochs{args.epochs}_{date_str}_{hour_str}"

    wandb.init(
        project="vit4cot_seg",
        name=run_name,
        config=args,
    )
    return wandb.run

def main():
    args = parse_args()
    # set_seed(args.seed)
    init_wandb(args)
    device = args.device

    train_loader = make_train_loader(args.data_dir, args.batch_size)
    val1_loader = make_val_loader(args.val1_dir, args.batch_size)
    val2_loader = make_val_loader(args.val2_dir, args.batch_size)

    model = create_model(args.encoder_name, args.hidden_size, args.feature_dim, args.num_classes).to(device)

    # class_weight = torch.FloatTensor([0.8,0.8,0.6,0.87,0.8])
    # criterion= nn.CrossEntropyLoss(weight=class_weight,label_smoothing=0.1).to(device)

    class_weight = torch.FloatTensor([0.85, 1.06, 0.15])
    criterion= nn.CrossEntropyLoss(weight=class_weight,label_smoothing=0.1).to(device)

    if args.optimizer == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer:{args.optimizer}")

    train(model, train_loader, val1_loader, val2_loader, criterion, optimizer,args)

if __name__ == '__main__':
    main()
