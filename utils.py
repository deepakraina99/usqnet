import matplotlib.pyplot as plt
import collections
import torch
from torch.utils.data.sampler import WeightedRandomSampler
import time, copy
import pickle
import seaborn as sn
import pandas as pd
import numpy as np

from sklearn.metrics import confusion_matrix, classification_report

def get_weighted_sampler(train_labels, cls_labels):
    counter=collections.Counter(train_labels)
    od_counter = collections.OrderedDict(sorted(counter.items()))
    class_count = [od_counter[i] for i in range(len(cls_labels))]
    class_weights = 1./torch.tensor(class_count, dtype=torch.float)
    train_labels = [int(i) for i in train_labels]
    target_list = torch.tensor(train_labels, dtype=torch.int64)
    # target_list = [i-1 for i in train_labels]
    class_weights_all = class_weights[target_list]
    weighted_sampler = WeightedRandomSampler(
        weights=class_weights_all,
        num_samples=len(class_weights_all),
        replacement=True
        )
    return weighted_sampler

# load_data('/home/mythra/Deepak/rus_ws/src/quality_assessment/usqnet/dataset/')

def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=100, fold=0, device='cuda', model_dir='/trained_models/'):
    since = time.time()

    val_acc_history, val_loss_history = [], []
    train_acc_history, train_loss_history = [], []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
#         print('-' * 10)

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
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    loss = criterion(outputs[0], labels)
                    _, preds = torch.max(outputs[0], 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            # print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
#                 print('improvement detected, saving model...')
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), model_dir+'/model_f{}.pth'.format(fold))                
            
            if phase == 'val':
                val_loss_history.append(epoch_loss)
                val_acc_history.append(epoch_acc)
                print('\rEpoch {}/{} {}id Loss: {:.4f} Acc: {:.4f}'.format(epoch+1, num_epochs, phase, epoch_loss, epoch_acc), end="")

            if phase == 'train':
                train_loss_history.append(epoch_loss)
                train_acc_history.append(epoch_acc)
                print('\rEpoch {}/{} {} Loss: {:.4f} Acc: {:.4f}'.format(epoch+1, num_epochs, phase, epoch_loss, epoch_acc), end="")
    
    time_elapsed = time.time() - since
    print('\nTraining complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    
    return model, (train_loss_history, train_acc_history, val_loss_history, val_acc_history)

def test_model(model, dataloader, class_labels, device='cuda', save_dir='/outputs/'):
    y_pred, y_true = [], []
    model.eval()   # Set model to evaluate mode

    # Iterate over data.
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        with torch.no_grad():
            outputs = model(inputs)
            output = torch.max(outputs[0], 1)[1]
            output = output.tolist()
            y_pred.extend(output) # Save Prediction
            true_label = labels.data.cpu().numpy()
            y_true.extend(true_label) # Save Truth
    score_dump = [y_true, y_true]
    # Build confusion matrix
    plt.figure()
    cfm=confusion_matrix(y_true, y_pred)
    print(cfm)
    FP = cfm.sum(axis=0) - np.diag(cfm) 
    FN = cfm.sum(axis=1) - np.diag(cfm)
    TP = np.diag(cfm)
    TN = cfm.sum() - (FP + FN + TP)
    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    # Specificity or true negative rate
    TNR = TN/(TN+FP)
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    # F1 Score for each class
    F1 = 2*((PPV*TPR)/(PPV+TPR))
    # Overall accuracy for each class
    ACC = (TP+TN)/(TP+FP+FN+TN)
    
    # Average for all classes (Need to be changed according to size of dataset)
    class_length_weights = np.array([0.2558647026732133, 0.15602836879432624, 0.11238406983087834, 0.2645935624659029, 0.21112929623567922])

    avg_ppv = np.dot(PPV,class_length_weights)
    avg_tpr = np.dot(TPR,class_length_weights)
    avg_tnr = np.dot(TNR,class_length_weights)
    avg_acc = np.dot(ACC,class_length_weights)
    avg_f1 = np.dot(F1,class_length_weights)
    AVG = np.array([avg_tpr, avg_tnr, avg_f1, avg_ppv, avg_acc])
    
    eval_mat = np.array([TPR,TNR,F1,PPV,ACC]).T
    eval_mat = np.append(eval_mat,[AVG],axis=0)
    
    df = pd.DataFrame(eval_mat*100, columns = ['Sens','Spec','F1','Prec','Acc'], index=class_labels+['Avg.'])
    print(df)
