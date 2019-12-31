import torch
from datetime import datetime
from time import time
from pathlib import Path
from tensorboardX import SummaryWriter
class Trainer:

    def __init__(self, device, epochs, dataloader, net, optimizer, lr_scheduler, loss, best_model_save_dir='./savedmodel', model_save_dir='./savedmodel', 
                logger=None, ts_dir='./runs', valloader=None, val_per_epochs=0):
        
        self.device = device
        self.epochs = epochs
        self.dataloader = dataloader
        self.net = net
        self.optimizer = optimizer
        self.loss = loss
        self.lr_scheduler = lr_scheduler
        self.best_model_save_dir = Path(best_model_save_dir)
        if not self.best_model_save_dir.exists():
            self.best_model_save_dir.mkdir(parents=True)
        self.model_save_dir = Path(model_save_dir)
        if not self.model_save_dir.exists():
            self.model_save_dir.mkdir(parents=True)

        self.writer =  SummaryWriter(ts_dir) if ts_dir!='' else None
        
        self.logger = logger
        self.valloader = valloader
        self.val_per_epochs = val_per_epochs
        self.best = None

    
    def train(self):
        for epoch in range(self.epochs):
            if self.valloader and (epoch+1)%self.val_per_epochs == 0:
                result = self.val_epochs(epoch)
                mode = 'Validation'
                
                #Save the net
                now = datetime.now().strftime('%m_%d-%H_%M')
                filename = self.model_save_dir/f'{self.net.__class__.__name__}_{now}.pth'
                torch.save(self.net.state_dict(), filename)
            else:
                result = self.train_epoch(epoch+1)
                mode = 'Training'

            # print info
            info = mode + f': epoch:{epoch}, '
            for k, in result.keys():
                info += f'{k}:{result[k]} '
            print(info)

            # log info 
            if self.logger: self.logger.info(info)

            # Save Best Model
            if self._check_best(result):
                filename = self.best_model_save_dir/f'{self.net.__class__.__name__}_best.pth'
                torch.save(self.net.state_dcit(), filename)

    def train_epoch(self, epoch):
        self.net.train()
        total_loss = 0
        tic = time()
        for idx, (left, right, disp) in enumerate(self.dataloader):
            tic = time()
            itr = epoch * len(self.dataloader) + idx
            left, right, disp = left.to(self.device), right.to(self.device), disp.to(self.device)
            outputs = self.net(left, right)

            # compute loss and backwards
            self.optimizer.zero_grad()
            loss = self.loss(outputs, disp)
            
            if self.lr_scheduler:
                lr = self.lr_scheduler(itr)
                self.optimizer.lr = lr
            self.optimizer.step()

            # print info
            info = f'Training, itr:{itr}, loss:{loss}, time:{time()-tic}'
            print(info)
            if self.logger: self.logger.info(info)
        
            if self.writer:
                self.writer.add_scalar('Training_loss',loss, itr)
                self.writer.add_scalar('Training_lr', lr, itr)
        result = {'loss':total_loss/idx, 'time':time()-tic}
        return result

    def val_epochs(self, epoch):
        self.net.eval()
        total_loss = 0
        tic = time()
        times = epoch // self.val_per_epochs - 1
        for idx, (left, right, disp) in enumerate(self.valloader):
            tic = time()
            itr = times * len(self.valloader) + idx
            left, right, disp = left.to(self.device), right.to(self.device), disp.to(self.device)
            outputs = self.net(left, right)
            # compute loss 
            loss = self.loss(outputs, disp)
            # print info
            info = f'Validation, itr:{itr}, loss:{loss}, time:{time()-tic}'
            print(info)
            if self.logger: self.logger.info(info)
        
            if self.writer:
                self.writer.add_scalar('Validation_loss',loss, itr)
                
        result = {'loss':total_loss/idx, 'time':time()-tic}
        return result

    def _check_best(self, result):
        if not self.best:
            self.best = result['loss']
            return True
        if self.best > result['loss']:
            self.best = result['loss']
            return True
        else:
            return False