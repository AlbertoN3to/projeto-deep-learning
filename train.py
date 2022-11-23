import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchdata
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import time
import os
import copy
from models.maxvit_model import MaxVit_Model
from data_loader.data_loaders import AlzheimerDataLoader


SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False



def train_model(model, criterion, optimizer, scheduler, database, model_path, device, num_epochs=25):
    since = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    dataset_sizes = database.dataset_sizes
    dataloaders = database.data_loaders
    
    early_stopper = EarlyStopper(patience=3, min_delta=0.01)
    y_loss = {}  # loss history
    y_loss['train'] = []
    y_loss['val'] = []
    y_err = {}
    y_err['train'] = []
    y_err['val'] = []
    

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:                
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).detach().cpu().numpy()
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = float(running_corrects) / dataset_sizes[phase]
            # epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            y_loss[phase].append(epoch_loss)
            y_err[phase].append(epoch_acc)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' :
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), os.path.join(os.sep.join(model_path.split(os.sep)[:-1]),str(epoch)+"_"+model_path.split(os.sep)[-1]) )
        

                if early_stopper.early_stop(running_loss):             
                    break

        print()
    
    plt.plot(y_err["train"],'-o')
    plt.plot(y_err["val"],'-o')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['Train','Valid'])
    plt.title('Train vs Valid Accuracy')

    plt.savefig("acc.png")
    plt.clf()

    plt.plot(y_loss["train"],'-o')
    plt.plot(y_loss["val"],'-o')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Train','Valid'])
    plt.title('Train vs Valid Loss')

    plt.savefig("loss.png")
   
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')
    # load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), model_path)
    return model

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(2)  # pause a bit so that plots are updated


# def visualize_model(model, num_images=6):
#     was_training = model.training
#     model.eval()
#     images_so_far = 0
#     fig = plt.figure()

#     with torch.no_grad():
#         for i, (inputs, labels) in enumerate(dataloaders['val']):
#             inputs = inputs.to(device)
#             labels = labels.to(device)

#             outputs = model(inputs)
#             _, preds = torch.max(outputs, 1)

#             for j in range(inputs.size()[0]):
#                 images_so_far += 1
#                 ax = plt.subplot(num_images//2, 2, images_so_far)
#                 ax.axis('off')
#                 ax.set_title(f'predicted: {class_names[preds[j]]} actual: {class_names[labels[j]]} ')
#                 imshow(inputs.cpu().data[j])

#                 if images_so_far == num_images:
#                     model.train(mode=was_training)
#                     return
#         model.train(mode=was_training)


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MaxVit_Model()
    model = model.model
    model.to(device)
    database = AlzheimerDataLoader(args.data_dir, args.test_dir, args.batch_size, num_workers=args.num_workers)
    
    criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    model_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=args.optim_step_size, gamma=args.optim_gamma)

    model = train_model(model, criterion, optimizer, model_lr_scheduler, database, args.model_path, device,
                           num_epochs=25)

    # visualize_model(model)
    
if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Train Alzheimer model')
    args.add_argument('--model_path', default="saved/best.pt", type=str,
                      help='path to save checkpoint (default: saved/best.pt)')
    args.add_argument('--data_dir', default="database/Alzheimer_Dataset/train", type=str,
                      help='path to training dataset (default: database/Alzheimer_Dataset/train)')
    args.add_argument('--test_dir', default="database/Alzheimer_Dataset/test", type=str,
                      help='path to test dataset (default: database/Alzheimer_Dataset/test)')
    args.add_argument('--batch_size', default=32, type=int,
                      help='batch size (default: 32)')
    args.add_argument('--lr', default=0.001, type=float,
                      help='learning rate (default: 0.001)')
    args.add_argument('--optim_step_size', default=4, type=int,
                      help='optimizer step size (default: 7)')
    args.add_argument('--optim_gamma', default=0.1, type=float,
                      help='optimizer gamma (default: 0.1)')
    args.add_argument('--num_workers', default=2, type=int,
                      help='number of workers (default: 2)')

    
    main(args.parse_args())
