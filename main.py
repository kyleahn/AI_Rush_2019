import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import argparse
import pathlib
from model import Resnet
import nsml
import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from dataloader import AIRushDataset
from nsml import DATASET_PATH

def to_np(t):
    return t.cpu().detach().numpy()

def bind_model(model_nsml):
    def save(dir_name, **kwargs):
        save_state_path = os.path.join(dir_name, 'state_dict.pkl')
        state = {
                    'model': model_nsml.state_dict(),
                }
        torch.save(state, save_state_path)

    def load(dir_name):
        save_state_path = os.path.join(dir_name, 'state_dict.pkl')
        state = torch.load(save_state_path)
        model_nsml.load_state_dict(state['model'])
        
    def infer(test_image_data_path, test_meta_data_path):
        # DONOTCHANGE This Line
        test_meta_data = pd.read_csv(test_meta_data_path, delimiter=',', header=0)
        
        input_size=224 # you can change this according to your model.
        batch_size=200 # you can change this. But when you use 'nsml submit --test' for test infer, there are only 200 number of data.
        device = 0
        
        dataloader = DataLoader(
                        AIRushDataset(test_image_data_path, test_meta_data, label_path=None,
                                      transform=transforms.Compose([transforms.Resize((input_size, input_size)), transforms.ToTensor()])),
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=0,
                        pin_memory=True)
        
        model_nsml.to(device)
        model_nsml.eval()
        predict_list = []
        for batch_idx, image in enumerate(dataloader):
            image = image.to(device)
            output = model_nsml(image).double()
            
            output_prob = F.softmax(output, dim=1)
            predict = np.argmax(to_np(output_prob), axis=1)
            predict_list.append(predict)
                
        predict_vector = np.concatenate(predict_list, axis=0)
        return predict_vector # this return type should be a numpy array which has shape of (138343)

    # DONOTCHANGE: They are reserved for nsml
    nsml.bind(save=save, load=load, infer=infer)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='parser')
    # DONOTCHANGE: They are reserved for nsml
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--iteration', type=str, default='0')
    parser.add_argument('--pause', type=int, default=0)
    
    # custom args
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--output_size', type=int, default=350) # Fixed
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=2.5e-4)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda:0")
    print('Train with %d GPU(s)' % (torch.cuda.device_count()))
    device_ids = list(range(torch.cuda.device_count()))

    assert args.input_size == 224
    model = torch.nn.DataParallel(Resnet(args.output_size), device_ids=device_ids)

    optimizer = optim.Adam(model.parameters(), args.learning_rate)
    criterion = nn.CrossEntropyLoss() #multi-class classification task

    model = model.to(device)
    model.train()

    # DONOTCHANGE: They are reserved for nsml
    bind_model(model)
    if args.pause:
        nsml.paused(scope=locals())
    if args.mode == "train":
        # Warning: Do not load data before this line
        image_dir = os.path.join(DATASET_PATH, 'train', 'train_data', 'images') 
        train_label_path = os.path.join(DATASET_PATH, 'train', 'train_label') 
        train_meta_path = os.path.join(DATASET_PATH, 'train', 'train_data', 'train_with_valid_tags.csv')
        train_meta_data = pd.read_csv(train_meta_path, delimiter=',', header=0)
        
        full_dataset = AIRushDataset(image_dir, train_meta_data, label_path=train_label_path, 
                      transform=transforms.Compose([transforms.Resize((args.input_size, args.input_size)), transforms.ToTensor()]))
        
        train_size = int(0.95 * len(full_dataset))
        test_size = len(full_dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

        print('Number of Train data :', len(train_dataloader))
        print('Number of Test data :', len(test_dataloader))
        for epoch_idx in range(1, args.epochs + 1):
            epoch_start_time = time.time()

            (test_image, test_tags) = next(iter(test_dataloader))
            for batch_idx, (image, tags) in enumerate(train_dataloader):
                niter = epoch_idx * len(train_dataloader) + batch_idx

                # train
                model.train()

                optimizer.zero_grad()
                image = image.to(device); tags = tags.to(device)
                output = model(image).double()
                loss = criterion(output, tags)
                loss.backward()
                optimizer.step()

                output_prob = F.softmax(output, dim=1)
                predict_vector = np.argmax(to_np(output_prob), axis=1)
                label_vector = to_np(tags)
                bool_vector = predict_vector == label_vector
                train_accuracy = bool_vector.sum() / len(bool_vector)
                train_loss = loss.item()

                # test
                model.eval()

                test_image = test_image.to(device); test_tags = test_tags.to(device)
                output = model(test_image).double()
                loss = criterion(output, test_tags)

                output_prob = F.softmax(output, dim=1)
                predict_vector = np.argmax(to_np(output_prob), axis=1)
                label_vector = to_np(test_tags)
                bool_vector = predict_vector == label_vector
                test_accuracy = bool_vector.sum() / len(bool_vector)
                test_loss = loss.item()
                
                # save model
                if batch_idx % args.log_interval == 0:
                    nsml.save(niter)
                # print log
                print('[{}/{}][{}/{}]: Train Loss {:2.4f} / Train Acc {:2.4f} / Test Loss {:2.4f} / Test Acc {:2.4f}'.format(epoch_idx,
                                                            args.epochs,
                                                            batch_idx,
                                                            len(train_dataloader),
                                                            train_loss,
                                                            train_accuracy,
                                                            test_loss,
                                                            test_accuracy))
                # write loss
                nsml.report(
                    summary=True,
                    step=niter,
                    scope=locals(),
                    **{
                    "train__loss": train_loss,
                    "train__accuracy": train_accuracy,
                    "test__loss": test_loss,
                    "test__accuracy": test_accuracy,
                    })
            print('Time :', time.time() - epoch_start_time)
