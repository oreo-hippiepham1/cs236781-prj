import pytorch_lightning as pl
import torch
import torchvision
import math
from .my_model import prepare_retina


class RetinaNetModule(pl.LightningModule):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, images, targets=None):
        return self.model(images, targets)


    def training_step(self, batch, batch_idx):
        images, targets = batch 
        losses = self.forward(images, targets)
        loss = losses["classification"] + losses["bbox_regression"]
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
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







