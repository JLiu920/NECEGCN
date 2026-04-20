import logging, torch, pickle, numpy as np
from tqdm import tqdm
from time import time
from . import utils as U
from .initializer import Initializer


class Processor(Initializer):
    def train(self, epoch):
        self.model.train()
        timer = dict(start_time=time(),curr_time=time(),end_time=time(),dataloader=0.001, model=0.001, statistics=0.001)
        num_top1, num_sample = 0, 0
        train_iter = self.train_loader if self.no_progress_bar else tqdm(self.train_loader, dynamic_ncols=True)
        for num, (x, y, _) in enumerate(train_iter):
            self.optimizer.zero_grad()
            x = x.float().to(self.device)
            y = y.long().to(self.device)
            timer['dataloader'] += time()-timer['curr_time']
            timer['curr_time'] = time()
            out, _ = self.model(x)
            loss = self.loss_func(out, y)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.global_step += 1

            timer['model'] += time()-timer['curr_time']
            timer['curr_time'] = time()
            num_sample += x.size(0)
            reco_top1 = out.max(1)[1]
            num_top1 += reco_top1.eq(y).sum().item()
            lr = self.optimizer.param_groups[0]['lr']
            if self.scalar_writer:
                self.scalar_writer.add_scalar('learning_rate', lr, self.global_step)
                self.scalar_writer.add_scalar('train_loss', loss.item(), self.global_step)
            if self.no_progress_bar:
                logging.info('Epoch: {}/{}, Batch: {}/{}, Loss: {:.4f}, LR: {:.4f}'.format(
                    epoch+1, self.max_epoch, num+1, len(self.train_loader), loss.item(), lr
                ))
            else:
                train_iter.set_description('Loss: {:.4f}, LR: {:.4f}'.format(loss.item(), lr))
            timer['statistics'] += time()-timer['curr_time']
            timer['curr_time'] = time()
        
        timer['total'] = time()-timer['start_time']
        train_acc = num_top1 / num_sample
        if self.scalar_writer:
            self.scalar_writer.add_scalar('train_acc', train_acc, self.global_step)
        logging.info('Epoch: {}/{}, Training accuracy: {:d}/{:d}({:.2%}), Training time: {:.2f}s'.format(
            epoch+1, self.max_epoch, num_top1, num_sample, train_acc, timer['total']
        ))
        logging.info('Dataloader: {:.2f}s({:.1f}%), Network: {:.2f}s({:.1f}%), Statistics: {:.2f}s({:.1f}%)'.format(
            timer['dataloader'], timer['dataloader']/timer['total']*100, timer['model'], timer['model']/timer['total']*100,
            timer['statistics'], timer['statistics']/timer['total']*100 
        ))
        logging.info('')

    def eval(self, save_score=False):
        self.model.eval()
        start_eval_time = time()
        score = {}

        with torch.no_grad():
            num_top1, num_top5 = 0, 0
            num_sample, eval_loss = 0, []
            cm = np.zeros((self.num_class, self.num_class))
            eval_iter = self.eval_loader if self.no_progress_bar else tqdm(self.eval_loader, dynamic_ncols=True)
            true_label = []
            pred_label = []
            features=[]
            for num, (x, y, name) in enumerate(eval_iter):
                x = x.float().to(self.device)
                y = y.long().to(self.device)
                out, feature = self.model(x)
                loss = self.loss_func(out, y)
                eval_loss.append(loss.item())
                if self.args.evaluate:
                    predicted1 = torch.argmax(out.data, 1)
                    lab = np.array(y.cpu())
                    pre = np.array(predicted1.cpu())
                    fea = np.array(feature.cpu())
                    features.append(fea)
                    true_label = np.concatenate((true_label, lab))
                    pred_label = np.concatenate((pred_label, pre))
                if save_score:
                    for n,c in zip(name,out.detach().cpu().numpy()):
                        score[n] = c
                num_sample += x.size(0)
                reco_top1 = out.max(1)[1]
                num_top1 += reco_top1.eq(y).sum().item()
                reco_top5 = torch.topk(out,5)[1]
                num_top5 += sum([y[n] in reco_top5[n,:] for n in range(x.size(0))])
                for i in range(x.size(0)):
                    cm[y[i], reco_top1[i]] += 1
                if self.no_progress_bar and self.args.evaluate:
                    logging.info('Batch: {}/{}'.format(num+1, len(self.eval_loader)))
        if self.args.evaluate:
            features = np.vstack(features)
            np.save('./feature.npy', features)
            np.save('./true_label_NTUCsub.npy', true_label)
            np.save('./pred_label_NTUCsub.npy', pred_label)
        acc_top1 = num_top1 / num_sample
        acc_top5 = num_top5 / num_sample
        eval_loss = sum(eval_loss) / len(eval_loss)
        eval_time = time() - start_eval_time
        eval_speed = len(self.eval_loader) * self.eval_batch_size / eval_time / len(self.args.gpus)
        logging.info('Top-1 accuracy: {:d}/{:d}({:.2%}), Top-5 accuracy: {:d}/{:d}({:.2%}), Mean loss:{:.4f}'.format(
            num_top1, num_sample, acc_top1, num_top5, num_sample, acc_top5, eval_loss
        ))
        logging.info('Evaluating time: {:.2f}s, Speed: {:.2f} sequnces/(second*GPU)'.format(
            eval_time, eval_speed
        ))
        logging.info('')
        if self.scalar_writer:
            self.scalar_writer.add_scalar('eval_acc', acc_top1, self.global_step)
            self.scalar_writer.add_scalar('eval_loss', eval_loss, self.global_step)

        torch.cuda.empty_cache()
        if save_score:
            return acc_top1, acc_top5, score
        else:
            return acc_top1, acc_top5, cm

    def start(self):
        start_time = time()
        if self.args.evaluate:
            if self.args.debug:
                logging.warning('Warning: Using debug setting now!')
                logging.info('')
            logging.info('Loading evaluating model ...')
            checkpoint = U.load_checkpoint(self.args.work_dir, self.model_name)
            if checkpoint:
                self.model.module.load_state_dict(checkpoint['model'])
            logging.info('Successful!')
            logging.info('')
            logging.info('Starting evaluating ...')
            self.eval()
            logging.info('Finish evaluating!')

        else:
            start_epoch = 0
            best_state = {'acc_top1':0, 'acc_top5':0, 'cm':0, 'best_epoch':0}
            if self.args.resume:
                logging.info('Loading checkpoint ...')
                checkpoint = U.load_checkpoint(self.args.work_dir)
                self.model.module.load_state_dict(checkpoint['model'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.scheduler.load_state_dict(checkpoint['scheduler'])
                start_epoch = checkpoint['epoch']
                best_state.update(checkpoint['best_state'])
                self.global_step = start_epoch * len(self.train_loader)
                logging.info('Start epoch: {}'.format(start_epoch+1))
                logging.info('Best accuracy: {:.2%}'.format(best_state['acc_top1']))
                logging.info('Successful!')
                logging.info('')
            logging.info('Starting training ...')
            for epoch in range(start_epoch, self.max_epoch):
                self.train(epoch)
                is_best = False
                if (epoch + 1) % 1 == 0:
                    logging.info('Evaluating for epoch {}/{} ...'.format(epoch+1, self.max_epoch))
                    acc_top1, acc_top5, cm = self.eval()
                    if acc_top1 > best_state['acc_top1']:
                        is_best = True
                        best_state.update({'acc_top1':acc_top1, 'acc_top5':acc_top5, 'cm':cm, 'best_epoch':epoch+1})
                logging.info('Saving model for epoch {}/{} ...'.format(epoch+1, self.max_epoch))
                U.save_checkpoint(
                    self.model.module.state_dict(), self.optimizer.state_dict(), self.scheduler.state_dict(),
                    epoch+1, best_state, is_best, self.args.work_dir, self.save_dir, self.model_name
                )
                logging.info('Best top-1 accuracy: {:.2%}@{}th epoch, Total time: {}'.format(
                    best_state['acc_top1'], best_state['best_epoch'], U.get_time(time()-start_time)
                ))
                logging.info('')
            np.savetxt('{}/cm.csv'.format(self.save_dir),cm,fmt="%s",delimiter=",")
            logging.info('Finish training!')
            logging.info('')

    def extract(self):
        logging.info('Starting extracting ...')
        if self.args.debug:
            logging.warning('Warning: Using debug setting now!')
            logging.info('')
        logging.info('Loading evaluating model ...')
        checkpoint = U.load_model(self.args.work_dir, self.model_name)
        if checkpoint:
            self.cm = checkpoint['best_state']['cm']
            self.model.module.load_state_dict(checkpoint['model'])
        logging.info('Successful!')
        logging.info('')
        acc_top1, acc_top5, score = self.eval(save_score=True)
        if not self.args.debug:
            U.create_folder('./visualization')
            save_path = self.args.work_dir + '/' + 'score.npy'
            with open(save_path, 'wb') as f:
                pickle.dump(score, f)
        logging.info('Finish extracting!')
        logging.info('')
