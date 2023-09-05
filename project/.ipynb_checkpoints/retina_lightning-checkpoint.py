import pytorch_lightning as pl
import torch
import torchvision
import math
from .my_model import prepare_retina


class RetinaNetModule(pl.LightningModule):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model
        self.train_step_outputs = {'loss': [], 'accuracy': [], 'mod_accuracy': []}
        self.valid_step_outputs = {'loss': [], 'accuracy': [], 'mod_accuracy': []}
        self.test_outputs = {'loss': [], 'accuracy': [], 'mod_accuracy': []}

    def forward(self, images, targets=None):
        return self.model(images, targets)


    def training_step(self, batch, batch_idx):
        images, targets = batch 
        print('TRAINING')
        losses = self.forward(images, targets)
        loss = losses["classification"] + losses["bbox_regression"]
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.train_step_outputs['loss'].append(loss)
        
        return loss
    
    def on_train_epoch_end(self):
        avg_loss = torch.stack(self.train_step_outputs['loss']).mean()  
        
        #self.log_dict({"train_loss": avg_loss, "train_acc": acc, 'train_mod_acc': mod_acc})
        print(f"TRAINING ---- {avg_loss=} ----")

    def validation_step(self, batch, batch_idx):
        print(f'{len(batch)=}')
        print(f'{len(batch[0])=}')
        print(f'{batch[0]=}')
        images, targets = batch
        print('VALIDATION')
        self.model.train()
        losses = self.forward(images, targets)
        val_loss = losses["classification"] + losses["bbox_regression"]  
        self.model.eval()
        return {'val_loss': val_loss}

    def on_validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('val_loss', avg_loss)

    def test_step(self, batch, batch_idx):
        images, targets = batch
        self.model.train()
        losses = self.forward(images, targets)
        test_loss = losses["classification"] + losses["bbox_regression"]  
        
        self.log('test_loss', test_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        self.model.eval()
        return {'test_loss': test_loss}
    
    
    def predict_step(self, batch, batch_idx):
        images = batch
        outputs = self.forward(images)
        return outputs

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=1e-3)
        return optimizer







