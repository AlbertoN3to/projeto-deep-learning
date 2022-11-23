import os
import torch
import argparse
import numpy as np
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay, classification_report
from models.maxvit_model import MaxVit_Model
from data_loader.data_loaders import AlzheimerDataLoader
from tqdm import tqdm
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt


# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


# def _save_confusion_matrix(matrix, classes, folder, name=''):
#     """Save the confusion matrix as .png

#     Args:
#         matrix (numpy.ndarray): Confusion matrix data.
#         matrix (list): List of classes.
#         matrix (str): Path to save confusion matrix file.
#         name (str): Custom file name posfix.

#     Return:
#         A list of dictionaries python compatible.

#     Raise:
#         None
#     """

#     disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=classes)
#     disp.plot(include_values=True, xticksrotation=90.0, cmap='Greens')
#     disp.figure.setfigwidth(12.8)
#     disp.figure.set_figheight(9.6)
#     plt.tight_layout()
#     save_path = os.path.join(folder, f'confusionmatrix{name}.png')
#     plt.savefig(save_path, figsize=(100,10))
#     plt.clf()


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MaxVit_Model()
    model.load_state_and_eval(args.model_path)
    model = model.model
    model.to(device)
    database = AlzheimerDataLoader(args.data_dir, args.test_dir, args.batch_size)

    criterion = torch.nn.CrossEntropyLoss()
    metric = accuracy

    total_loss = 0.0
    total_metrics = 0
    predictions = []
    ground_truth = []
    
    
    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(database.data_loaders["test"])):
            data, target = data.to(device), target.to(device)
            output = model(data)

            #
            # save sample images, or do something with output here
            #

            # computing loss, metrics on test set
            loss = criterion(output, target)
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size
            
            pred = torch.argmax(output, 1)
            predictions.extend(pred.cpu().numpy())
            ground_truth.extend(target.data.cpu().numpy())
            total_metrics += metric(output, target) * batch_size
    
    cf_matrix = confusion_matrix(ground_truth, predictions)
    df_cm = pd.DataFrame(cf_matrix, index = database.classes,
                         columns = database.classes)
    print(df_cm)
    plt.figure(figsize = (12,7))
    sn.heatmap(df_cm, annot=True, fmt='d')
    plt.ylabel("True Class"), 
    plt.xlabel("Predicted Class")
    plt.savefig('confusionmatrix.png')

    n_samples = len(database.data_loaders["test"].sampler)
    
    print("Loss:",total_loss / n_samples)
    print("Acc: ", total_metrics / n_samples)
    print(classification_report(ground_truth, predictions, digits=3))

    

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Test Alzheimer model')
    args.add_argument('--model_path', default="saved/best.pt", type=str,
                      help='path to latest checkpoint (default: saved/best.pt)')
    args.add_argument('--data_dir', default="database/Alzheimer_Dataset/train", type=str,
                      help='path to training dataset (default: database/Alzheimer_Dataset/train)')
    args.add_argument('--test_dir', default="database/Alzheimer_Dataset/test", type=str,
                      help='path to test dataset (default: database/Alzheimer_Dataset/test)')
    args.add_argument('--batch_size', default=32, type=str,
                      help='batch size (default: 32)')
    
    main(args.parse_args())
