# importing the libraries
import os
import glob
import pandas as pd
import argparse

# PyTorch libraries
import torch
import torch.nn as nn
from torch.nn import Linear, Sequential

# Pre-trained models
from torchvision.models import vgg16, VGG16_Weights

from MRIDataset import MRIDataset
from trainer import train_model
from Prediction import Evaluation
from Model import CNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_args_parser():
    parser = argparse.ArgumentParser("MRIDetection", add_help=False)


    parser.add_argument(
        "--MRIfolder_neuroflux",
        type=str,
        help="""MRI folder
        """,
    )

    parser.add_argument(
        "--checkpoint",
        default="./MRIDetection.pth",
        type=str,
        help="""checkpoint path
         """,
    )
    
    parser.add_argument(
        "--model_name",
        default="VGG16",
        type=str,
        help="""model name
         """,
    )

    return parser


def main(args):
    
    # DATA PREPARATION  
    # Load Path
    path = args.MRIfolder_neuroflux
    MRIPath = glob.glob(path + "/*/neuroflux*.jpg")

    # Select annnotations
    classes = [ _ for _ in os.listdir(path) if not _.startswith(".")]
    num_classes = len(classes)
    
    Str_label = [MRIPath[i].split('/')[-2] for i in range(len(MRIPath))]
    labels = [classes.index(_) for _ in Str_label]
      
    # Resample len of differents class to same len
    df = pd.DataFrame(labels , columns = ["labels"])

    class_1 = df[df.labels == 1]   
    class_2 = df[df.labels == 2]   

    new_class_0 = df[df.labels == 0].sample(len(class_2))
    new_class_2 = df[df.labels == 2].sample(len(class_2))
    new_class_3 = df[df.labels == 3].sample(len(class_2))
    new_class_4 = df[df.labels == 4].sample(len(class_2))
    
    df_labels = pd.concat([new_class_0, class_1, new_class_2, new_class_3, new_class_4])
    final_labels = df_labels.labels.tolist()
    index = list(df_labels.index.values)
    
    # Select images path
    final_MRIPath = [ MRIPath[i] for i in index]
    
    # Prepare Dataset
    dataset = MRIDataset(final_MRIPath , final_labels)
    num_data = len(dataset)

    n_val_test = int(num_data * 0.4)
    n_train = len(dataset) - n_val_test
    n_test, n_val =  int(num_data * 0.2),  int(num_data * 0.2)    
    
    
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
        dataset, (n_train, n_val, n_test)
    )
    
    # Train Model
    # Hyperparamters
    batch_size = 16
    num_epochs = 5
    
    # Build Data Loader
    # Define data loaders for training and testing data in this fold
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size
    )
    validation_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size
    )
    
    test_loader  = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size
    )
    if args.model_name == "VGG16":
        # loading the pretrained model
        model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        
        # Freeze model weights of the VGG-16 model.
        for param in model.parameters():
            param.requires_grad = False
    
        # Add a Linear layer to the classifier
        model.classifier[-1] = Sequential(Linear(4096, num_classes).to(device))
    
        # Train the model by updating the weights of the last layer
        for param in model.classifier[-1].parameters():
            param.requires_grad = True
            
    if args.model_name == "CNN":

        model = CNN(num_classes)
    # Move to default device
    model.to(device)

    # Hyperparamters model
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Initialize model or load checkpoint
    trainModel = True # False if only prediction

    if trainModel == True:
        model = train_model(
            train_loader, validation_loader, model, loss_function, optimizer, num_epochs
        )

        # Save checkpoint
        torch.save(model.state_dict(), args.checkpoint)
    else:
        # Load checkpoint
        model.load_state_dict(torch.load(args.checkpoint))
        
    # Evaluation Model
    # Prediction
    Evaluation(test_loader, model, classes)


if __name__ == "__main__":

    parser = argparse.ArgumentParser("MRIDetection", parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
