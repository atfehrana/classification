# importing the libraries
import matplotlib.pyplot as plt
from torchmetrics import Accuracy, Recall, Precision, Specificity, F1Score

# PyTorch libraries
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def Evaluation(test_loader, model, classes):

    y_predList = []
    y_trueList = []
    
    with torch.no_grad():
        model.eval()
          
        for imagesTest, labelsTest in test_loader:
            
            imagesTest, labelsTest = imagesTest.to(device), labelsTest.to(device)    
            # Prediction
            predTest = model(imagesTest)
            
            # Statistics between predictionn and true class
            accuracy = Accuracy(task="multiclass", num_classes=5, top_k=4)
            test_acc = accuracy(predTest, labelsTest) 
            
            recall = Recall(task="multiclass", average = "macro", num_classes=5)
            test_recall = recall(predTest, labelsTest)   
            
            precision = Precision(task="multiclass", average = "macro", num_classes=5)
            test_precision = precision(predTest, labelsTest)  
            
            specificity = Specificity(task="multiclass", average = "macro", num_classes=5)
            test_specificity = specificity(predTest, labelsTest)  
            
            f1score = F1Score(task="multiclass", num_classes=5)
            test_f1score= f1score(predTest, labelsTest)  
            
            predTest_tag = torch.log_softmax(predTest, dim = 1)
            _, predTest_label = torch.max(predTest_tag, dim = 1)   
            
            # accumulating running prediction 
            y_predList.append(predTest_label.cpu().numpy())
            # accumulating running label
            y_trueList.append(labelsTest.cpu().numpy())

    print(f'ACC {test_acc:.3f} | Precision : {test_precision:.3f} | Recall : {test_recall:.3f} | Specificity : {test_specificity:.3f} | F1Score : {test_f1score:.3f}   ')  
    
    # Example of predict phase of disease given an MRI scan
    plt.imshow(imagesTest[0][0])
    plt.title(classes[y_predList[0][0]])
    plt.show()
    
    
