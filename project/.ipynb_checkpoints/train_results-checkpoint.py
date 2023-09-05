from typing import List, NamedTuple, Dict

class BatchResult(NamedTuple):
    '''
    Represents the result of training for a single batch:
        - Loss
        - MAP
    '''
    loss: Dict[str, List[float]]
    mean_average_prediction: float
    

class EpochResult(NamedTuple):
    '''
    Represents the result of training for a single epoch:
        - loss per batch
        - MAP on dataset (train / test)
    '''
    losses: List[Dict[str, List[float]]]
    mean_average_prediction: float
    

class FitResult(NamedTuple):
    '''
    Represents the result of fitting a model for multiple epochs 
        given a training and valid (test) set
    Losses are for each batch and accuracies are per epoch
    '''
    num_epochs: int
    train_loss: List[Dict[str, List[float]]]
    train_map: List[Dict[str, List[float]]]
    test_loss: List[Dict[str, List[float]]]
    test_map: List[Dict[str, List[float]]]