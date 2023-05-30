# PyTorch libraries 
import torch
from torchmetrics import Accuracy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(
    train_loader, validation_loader, model, loss_function, optimizer, num_epochs
):
    accuracy_stats = {
        'train': [],
        "val": []
    }
    loss_stats = {
        'train': [],
        "val": []
    }
  
    y_predList = []
    y_trueList = []
    print("Training...")
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        #TRAINING    
        train_epoch_loss = 0
        train_epoch_acc = 0    
        model.train()
        for imagesTrain, labelsTrain in train_loader:
             
            imagesTrain = imagesTrain.to(device)
            labelsTrain = labelsTrain.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()        
            # forward pass
            predTrain = model(imagesTrain) 
            # defining loss
            train_loss = loss_function(predTrain, labelsTrain)
            

            # defining accuracy
            accuracy = Accuracy(task="multiclass", num_classes=5, top_k=4)
            train_acc = accuracy(predTrain, labelsTrain)  
            
            # computing gradients
            train_loss.backward()
            optimizer.step()  
            # accumulating running loss
            train_epoch_loss += train_loss.item()
            # accumulating running acc
            train_epoch_acc += train_acc.item()
            
        # VALIDATION
        with torch.no_grad():
            model.eval()
            val_epoch_loss = 0
            val_epoch_acc = 0
            
            for imagesVal, labelsVal in validation_loader:
                imagesVal, labelsVal = imagesVal.to(device), labelsVal.to(device)          
                predVal = model(imagesVal)
                
                val_loss = loss_function(predVal, labelsVal)
                val_acc = accuracy(predVal, labelsVal) 
               
                val_epoch_loss += val_loss.item()
                val_epoch_acc += val_acc.item()           
                
                predVal_tag = torch.log_softmax(predVal, dim = 1)
                _, predVal_label = torch.max(predVal_tag, dim = 1)    

                # accumulating running prediction 
                y_predList.append(predVal_label.cpu().numpy())
                # accumulating running label
                y_trueList.append(labelsVal.cpu().numpy())

        
        
        loss_stats['train'].append(train_epoch_loss/len(train_loader))    
        accuracy_stats['train'].append(train_epoch_acc/len(train_loader))
        
        loss_stats['val'].append(val_epoch_loss/len(validation_loader))
        accuracy_stats['val'].append(val_epoch_acc/len(validation_loader))

        print(f'Epoch {epoch+0:02}: | Train Loss: {train_epoch_loss/len(train_loader):.5f} | Val Loss: {val_epoch_loss/len(validation_loader):.5f} | Train Acc: {train_epoch_acc/len(train_loader):.3f}| Val Acc: {val_epoch_acc/len(validation_loader):.3f}')    

    return model
