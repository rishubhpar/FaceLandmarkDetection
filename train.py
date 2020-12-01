import os
import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
from easydict import Easydict as edict 
from tensorboardX import SummaryWriter
from dataset.faceDB300W import *
from models.hg_model import *
from models.initializtion import *

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

def train_model(model, params, train_dataloader, val_dataloader, tf_writer):
    """ Function to train the landmark detection algorithm """
    train_losses = []
    train_mse = []
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'], momentum=params['momentum'], weight_decay = params['weight_decay'])
    best_acc = 0.0

    for e in range(params['epochs']):
        for i, data in enumerate(train_dataloader):
            image = data['image'].to(device)
            landmarks = data['landmark_map'].to(device)

            landmarks_predicted = model(image)
            output_hg1 = landmarks_predicted[0, ...]
            output_hg2 = landmarks_predicted[1, ...]
            output_hg3 = landmarks_predicted[2, ...]
            output_hg4 = landmarks_predicted[3, ...]

            alpha1, alpha2, alpha3, alpha4 = 0.25, 0.25, 0.25, 0.25

            loss = alpha1 * criterion(output_hg1, landmarks) + alpha2 * criterion(output_hg2, landmarks) + alpha3 * criterion(output_hg3, landmarks)
                   + alpha4 * criterion(output_hg4, landmarks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        acc = validate(model, params, val_dataloader, tf_writer)
        if (acc > best_acc):
            best_acc = acc
            save_checkpoint(model, "best_ckpt")



def train(params):
    """ Main function for training and evaluating the Network """

    student_hg_config = {'nStack': 4,
                         'nModules': 2,
                         'nFeats:': 256,
                         'downsample:', 4}

    model = hourglass(student_hg_config, 68)

    if torch.cuda.device_count() > 0:
        print("Using ", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model).cuda()

    # Initializing the model weights with xavier initialization
    model.apply(weights_init_xavier)

    # Creating test and train transforms for data augmentation
    train_transforms = []
    test_transforms = []

    train_dataset = torch.utils.data.DataLoader((faceDB300W(params['dataset_dir'], 'train', train_transforms)), batch_size = params['batch_size'], shuffle=True,
                                                num_workers = params['num_workers'], pin_memory = True, drop_last = True)

    val_dataset = torch.utils.data.DataLoader((faceDB300W(params['dataset_dir'], 'test', test_transforms)), batch_size = params['batch_size'], shuffle=True,
                                                num_workers = params['num_workers'], pin_memory = True, drop_last = True)


    tf_writer = SummaryWriter(log_dir = os.path.join(params['log_dir'], params['config_name']))

    train_model(model, params, train_transforms, tf_writer)


def main():
    params = {}
    params['lr'] = 0.0001
    params['momentum'] = 0.9
    params['weight_decay'] = 1e-4
    params['batch_size'] = 16
    params['num_workers'] = 4
    params['num_epochs'] = 5
    params['log_dir'] = './logs'
    params['dataset_dir'] = './data/300W'
    params['config_name'] = 'experiment'

    train(params)

if __name__ == "__main__":
    main()
    
