import os
import abc
import sys
import torch
import torch.nn as nn
import tqdm.auto
from torch import Tensor
import numpy as np
from typing import Any, Tuple, Callable, Optional, cast
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from .train_results import FitResult, BatchResult, EpochResult
#from classifier import Classifier

from sklearn.metrics import average_precision_score
from torchvision.ops import box_iou


class Trainer(abc.ABC):
    '''
    A class abstracting various tasks of training models
    
    Provides methods at multiple levels of granularity:
        - Multiple epochs (fit)
        - Single epoch (train_epoch / test_epoch)
        - Single batch (train_batch / test_batch)
    '''
    
    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.device = device
        
        if self.device:
            model = model.to(self.device)
            
    def fit(
        self,
        dl_train: DataLoader,
        dl_test: DataLoader,
        num_epochs: int,
        checkpoints: str = None,
        early_stopping: int = None,
        print_every: int = 1,
        **kw
    ) -> FitResult:
        
        actual_num_epochs = 0
        epochs_without_improvement = 0
        
        train_loss, train_map, test_loss, test_map = [], [], [], [] # initialize metrics
        best_map = None
        
        for epoch in range(num_epochs):
            verbose = False
            if print_every > 0 and (
                epoch % print_every == 0 or epoch == num_epochs - 1
            ):
                verbose = True
            self._print(f'--- EPOCH {epoch+1}/{num_epochs} ---', verbose)
            
            # TRAIN
            train_result = self.train_epoch(dl_train, **kw) # EpochResult obj: losses: List[float], map: float        

            train_loss += [l.item() for l in train_result.losses]
            train_map.append(train_result.mean_average_prediction)
            
            
            #TEST
            test_result = self.test_epoch(dl_test, **kw)
            test_loss += [l.item() for l in test_result.losses]
            test_acc.append(test_result.mean_average_prediction)
            
            # adds Ten
            # self.writer.add_scalar('Test Loss', np.mean([l.item()for l in train_result.losses]), epoch)
            # self.writer.add_scalar('Test Accuracy', test_result.accuracy, epoch)
            
            # Closing tensorboard writer
            #self.writer.close()
            
            actual_num_epochs += 1
            
            # EARLY STOPPING
            if best_map is None or test_result.map > bets_map:
                best_map = test_result.mean_average_prediction
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if early_stopping is not None and epochs_without_improvement >= early_stopping:
                    print(f"---- Stopping training after {epochs_without_improvement=} ----")
                    break
                    
            
            # SAVING CHECKPOINTS
            if checkpoints is not None and best_acc is not None and test_result.accuracy > best_acc:
                self.save_checkpoint(checkpoints)
            

        return FitResult(num_epochs, train_loss, train_map, test_loss, test_map)
    
    
    def save_checkpoint(self, cp_filename):
        torch.save(self.model, cp_filename)
        print(f"\n*** Saved checkpoint {cp_filename}")
    
    
    def train_epoch(self, dl_train: DataLoader, **kw) -> EpochResult:
        '''
        Train once over a training set (single epoch)
        '''
        self.model.train(True)
        return self._foreach_batch(dl_train, self.train_batch, **kw)
    
    def test_epoch(self, dl_test: DataLoader, **kw) -> EpochResult:
        '''
        Evaluate once over test / valid set (single epoch)
        '''
        self.model.train(True)
        return self._foreach_batch(dl_test, self.test_batch, **kw)
    
    @abc.abstractmethod
    def train_batch(self, batch) -> BatchResult:
        '''
        Run a single batch forward through model - calculate loss - back prop - update weight
        '''
        raise NotImplementedError() # to be overriden
        
    @abc.abstractmethod
    def test_batch(self, batch) -> BatchResult:
        raise NotImplementedError()
        
    @staticmethod
    def _print(msg, verbose=True):
        if verbose:
            print(msg)
            
    @staticmethod
    def _foreach_batch(
        dl: DataLoader,
        forward_fn: Callable[[Any], BatchResult],
        verbose=True,
        max_batches=None,
    ) -> EpochResult:
        '''
        Evaluates the given forward function on batches from the given DL, and prints result
        '''
        losses = []
        num_correct = 0
        num_samples = len(dl.sampler)
        num_batches = len(dl.batch_sampler)
        
        if max_batches is not None:
            if max_batches < num_batches:
                num_batches = max_batches
                num_samples = num_batches * dl.batch_size
        
        if verbose:
            pbar_fn = tqdm.auto.tqdm
            pbar_file = sys.stdout
        else:
            pbar_fn = tqdm.tqdm
            pbar_file = open(os.devnull, 'w')
        
        pbar_name = forward_fn.__name__
        with pbar_fn(desc=pbar_name, total=num_batches, file=pbar_file) as pbar:
            dl_iter = iter(dl)
            for batch_idx in range(num_batches):
                data = next(dl_iter)
                batch_res = forward_fn(data)
                
                losses = batch_res.losses
                mean_average_prediction = batch_res.mean_average_prediction
                
                pbar.set_description(f'{pbar_name} ({batch_res.loss:.7f})')
                pbar.update()
                
                # losses["classification"] + losses["bbox_regression"]  
                
                losses_cls.append(losses['classification'])
                losses_reg.append(losses['bbox_regression'])
            
            avg_losses_cls = sum(losses_cls) / num_batches
            avg_losses_reg = sum(losses_reg) / num_batches
            #accuracy = 100.0 * np.sum(num_correct) / (num_samples * 20) # 20: n_classes
            pbar.set_description(
                f'{pbar_name} '
                f'(Avg Loss Cls {avg_losses_cls:.7f}, '
                f'(Avg Loss Reg {avg_losses_reg:.7f}, '
                f'MAP {mean_average_prediction:.7f})'
            )
        
        if not verbose:
            pbar_file.close()
        
        return EpochResult(losses=losses, mean_average_prediction=mean_average_prediction)

class ODTrainer(Trainer):
    def __init__(
        self,
        model: nn.Module,
        #loss_fn: nn.Module,
        optimizer: Optimizer,
        device: Optional[torch.device] = None
    ):
        super().__init__(model, device)
        self.optimizer = optimizer
        # self.loss_fn = loss_fn
        
    
    def compute_map(self, preds, targets, n_classes=7, iou_threshold=0.5):
        # Prepare predicted detections
        pred_boxes = torch.cat([pred['boxes'] for pred in predictions], dim=0)
        pred_labels = torch.cat([pred['labels'] for pred in predictions], dim=0)
        pred_scores = torch.cat([pred['scores'] for pred in predictions], dim=0)

        # Sort detections by scores in descending order
        sorted_indices = torch.argsort(pred_scores, descending=True)
        pred_boxes = pred_boxes[sorted_indices]
        pred_labels = pred_labels[sorted_indices]
        pred_scores = pred_scores[sorted_indices]
        
        # Prepare ground truth annotations
        gt_boxes = torch.cat([target['boxes'] for target in targets], dim=0)
        gt_labels = torch.cat([target['labels'] for target in targets], dim=0)

        # Compute IoU between predicted detections and ground truth annotations
        iou = box_iou(pred_boxes, gt_boxes)
        
        # Assign each predicted detection to the ground truth annotation with highest IoU above threshold
        assigned_gt_indices = iou.argmax(dim=1)
        assigned_gt_iou = iou.max(dim=1).values
        assigned_gt_indices[assigned_gt_iou < iou_threshold] = -1

        # Compute average precision (AP) for each class
        ap_scores = []
        for class_id in range(num_classes):
            class_pred_boxes = pred_boxes[pred_labels == class_id]
            class_pred_scores = pred_scores[pred_labels == class_id]
            class_assigned_gt_indices = assigned_gt_indices[pred_labels == class_id]

            # Sort predicted detections by scores in descending order
            class_sorted_indices = torch.argsort(class_pred_scores, descending=True)
            class_pred_boxes = class_pred_boxes[class_sorted_indices]
            class_assigned_gt_indices = class_assigned_gt_indices[class_sorted_indices]

            # Compute precision and recall values
            true_positives = torch.zeros(len(class_pred_boxes))
            false_positives = torch.zeros(len(class_pred_boxes))
            gt_instances = (class_assigned_gt_indices != -1).sum().item()

            for i, assigned_gt_index in enumerate(class_assigned_gt_indices):
                if assigned_gt_index != -1:
                    true_positives[i] = 1
                    assigned_gt_indices[assigned_gt_index] = -1  # Mark assigned ground truth as used
                else:
                    false_positives[i] = 1

            cumulative_true_positives = torch.cumsum(true_positives, dim=0)
            cumulative_false_positives = torch.cumsum(false_positives, dim=0)
            recall = cumulative_true_positives / gt_instances
            precision = cumulative_true_positives / (cumulative_true_positives + cumulative_false_positives + 1e-10)

            # Compute AP using precision-recall curve
            ap = average_precision_score(true_positives.numpy(), precision.numpy())
            ap_scores.append(ap)

        # Compute mean average precision (mAP)
        mAP = sum(ap_scores) / n_classes

        return mAP
    
    
    
    def train_batch(self, batch) -> BatchResult:
        image, targets = batch
        
        if self.device:
            image, targets = image.to(self.device), targets.to(self.device)
            
        self.model: Classifier
        batch_loss: float
        mean_average_prediction: int
        
        # setup
        self.optimizer.zero_grad()
        
        # forward
        losses = self.model(image, targets)
        
        # backward
        losses.backward()
        self.optimizer.step()
        
        # Calculate MAP
        with torch.no_grad():
            preds = self.model(image)
            mean_average_prediction = self.compute_map(preds, targets)
        
        return BatchResult(batch_loss, mean_average_prediction)
    
    
    def test_batch(self, batch) -> BatchResult:
        image, targets = batch
        
        if self.device:
            image, targets = image.to(self.device), targets.to(self.device)
            
        self.model: Classifier
        losses = {}
        mean_average_prediction: float
        
        with torch.no_grad():
            preds = self.model(image)     
            losses = self.model(image, targets)

            # Calculate accuracy for each class
            mean_average_prediction = self.compute_map(preds, targets)
        
        return BatchResult(losses, mean_average_prediction)
    