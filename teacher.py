from datetime import datetime
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
from models.basic_model import AVClassifier
from dataset.CremadDataset import CremadDataset
from dataset.VGGSoundDataset import VGGSound
from dataset.AVMNIST import AVMNIST
from utils.utils import setup_seed, weight_init, unimodal_outputs, evaluate, calculate_accuracy, update_model_with_OGM_GE

# Import configuration directly
import config

def train(model, device, train_loader, criterion, optimizer, epoch):
    """
    Train the model for one epoch using the provided data loader.
    Handles different modalities based on the configuration.
    """
    
    model.train()
    total_loss = 0
    total_loss_a = 0
    total_loss_v = 0
    
    total_correct = 0
    total_correct_a = 0
    total_correct_v = 0
    
    total_samples = 0

    for data in train_loader:
        # Unpack data
        audio, video, labels = data
        audio, video, labels = audio.to(device), video.to(device), labels.to(device)
        if config.dataset == 'AVMNIST':
            audio = audio.float()
        else:
            audio = audio.unsqueeze(1).float()
        video = video.float()
        optimizer.zero_grad()
        
        if config.modality == 'multimodal':
            af, vf, outputs = model(audio, video, return_features=True)
            
            out_a, out_v = unimodal_outputs(model, af, vf)

            loss_a = criterion(out_a, labels) 
            loss_v = criterion(out_v, labels)
            
            total_loss_a += loss_a.item()
            total_loss_v += loss_v.item() 
            
            total_correct_a += calculate_accuracy(out_a, labels)
            total_correct_v += calculate_accuracy(out_v, labels)    
            
        else:
            if config.modality == 'audio':
                outputs = model(audio, None)
                
            if config.modality == 'visual':
                outputs = model(None, video)
        
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        
        total_correct += calculate_accuracy(outputs, labels)
        total_samples += labels.size(0)
        
        loss.backward()
        
        if 'OGM' in config.modulation:
            # print("Using OGM modulation")
            if config.modality != 'multimodal':
                raise NotImplementedError("OGM modulation only works with multimodal")
            
            ratio_v, coeff_a, coeff_v = update_model_with_OGM_GE(out_a, out_v, labels, model, epoch)
            # print(f"ratio_v: {ratio_v}, ratio_a: {1/ratio_v}")
            # print(f"coeff_a: {coeff_a}, coeff_v: {coeff_v}")
            
        optimizer.step()
        
    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * (total_correct / total_samples)

    if config.modality == 'multimodal':
        avg_loss_a = total_loss_a / len(train_loader)
        avg_loss_v = total_loss_v / len(train_loader)
        accuracy_a = 100 * (total_correct_a / total_samples)
        accuracy_v = 100 * (total_correct_v / total_samples)
        return avg_loss_a, avg_loss_v, avg_loss, accuracy_a, accuracy_v, accuracy
    
    return avg_loss, accuracy



def main():
    """
    Main training and evaluation loop.
    Setup based on modality and dataset configurations.
    """
    setup_seed(config.random_seed)

    # Get the current timestamp for uniqueness
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Modify the best_model_path to include the timestamp
    writer_log_dir = f"runs/{config.ckpt_string}_{current_time}"

    # Initialize the TensorBoard writer based on the best_model_path (without the .pth)
    writer = SummaryWriter(log_dir=writer_log_dir)

    # Import and log hyperparameters from config.py
    hparams = config.get_hparams()  # Call the function to get hyperparameters
    writer.add_hparams(hparams, {})  # Log the hyperparameters

    device = torch.device(config.device)
    args = {
        'role': config.role,
        'modality': config.modality,
        'fusion_method': config.fusion_method,
        'dataset': config.dataset
    }
    
    model = AVClassifier(args['role'], args['modality'], args['fusion_method'], args['dataset']).to(device)
    model.apply(weight_init)

    if config.dataset == 'CREMAD':
        train_dataset = CremadDataset(mode='train')
        test_dataset = CremadDataset(mode='test')
    elif config.dataset == 'VGGSound':
        train_dataset = VGGSound(mode='train')
        test_dataset = VGGSound(mode='test')
    elif config.dataset == 'AVMNIST':
        train_dataset = AVMNIST(mode='train')
        test_dataset = AVMNIST(mode='test')
    else:
        raise NotImplementedError(f"Incorrect dataset name {config.dataset}")
    
    # Select a small subsample of 10 samples for debugging
    if config.DEBUG:
        print("\n WARNING: Testing on a small dataset \n")
        train_dataset = torch.utils.data.Subset(train_dataset, range(10))
        test_dataset = torch.utils.data.Subset(test_dataset, range(10))
        
    
    # train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    # test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)  
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    criterion = nn.CrossEntropyLoss()
    if config.optimiser == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9, weight_decay=config.weight_decay)
    elif config.optimiser == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay = config.weight_decay, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.lr_decay_step, gamma=config.lr_decay_ratio)
    
    best_accuracy = 0.0
    best_epoch = 0
    best_model_state_dict = None
    best_path_with_accuracy = None
    for epoch in tqdm(range(config.epochs)):                
        # Print the learning rate
        for param_group in optimizer.param_groups:
            print(f"\nLearning rate at the BEGINNING of {epoch+1}: {param_group['lr']}")
        
        if config.modality == 'multimodal':
            train_loss_a, train_loss_v, train_loss, train_accuracy_a, train_accuracy_v, train_accuracy = train(model, device, train_loader, criterion, optimizer, epoch)
            test_loss_a, test_loss_v, test_loss, test_accuracy_a, test_accuracy_v, test_accuracy = evaluate(model, device, test_loader, criterion)

            # Log all metrics (including audio and video) for this epoch
            writer.add_scalar('Loss/train_total', train_loss, epoch)
            writer.add_scalar('Loss/train_audio', train_loss_a, epoch)
            writer.add_scalar('Loss/train_video', train_loss_v, epoch)
            writer.add_scalar('Accuracy/train_total', train_accuracy, epoch)
            writer.add_scalar('Accuracy/train_audio', train_accuracy_a, epoch)
            writer.add_scalar('Accuracy/train_video', train_accuracy_v, epoch)

            writer.add_scalar('Loss/test_total', test_loss, epoch)
            writer.add_scalar('Loss/test_audio', test_loss_a, epoch)
            writer.add_scalar('Loss/test_video', test_loss_v, epoch)
            writer.add_scalar('Accuracy/test_total', test_accuracy, epoch)
            writer.add_scalar('Accuracy/test_audio', test_accuracy_a, epoch)
            writer.add_scalar('Accuracy/test_video', test_accuracy_v, epoch)

            print(f'Epoch {epoch+1}: \nTrain Loss\t: {train_loss:.4f}, Train Acc\t : {train_accuracy:.2f}%, \t--\t Test Loss\t: {test_loss:.4f}, Test Acc\t: {test_accuracy:.2f}%, \n'
                f'Audio Train Loss: {train_loss_a:.4f}, Audio Train Acc: {train_accuracy_a:.2f}%, \t--\t Audio Test Loss: {test_loss_a:.4f}, Audio Test Acc: {test_accuracy_a:.2f}%, \n'
                f'Video Train Loss: {train_loss_v:.4f}, Video Train Acc: {train_accuracy_v:.2f}%, \t--\t Video Test Loss: {test_loss_v:.4f}, Video Test Acc: {test_accuracy_v:.2f}%\n')
            
        else:
            train_loss, train_accuracy = train(model, device, train_loader, criterion, optimizer, epoch)
            test_loss, test_accuracy = evaluate(model, device, test_loader, criterion)

            # Log the metrics for unimodal training
            writer.add_scalar('Loss/train_epoch', train_loss, epoch)
            writer.add_scalar('Accuracy/train_epoch', train_accuracy, epoch)
            writer.add_scalar('Loss/test_epoch', test_loss, epoch)
            writer.add_scalar('Accuracy/test_epoch', test_accuracy, epoch)

            print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, '
                f'Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.2f}%\n')
        
        if test_accuracy >= best_accuracy:
            # Delete the previous best model if it exists
            if best_path_with_accuracy is not None:
                os.remove(best_path_with_accuracy)
            print(f"New best model: {test_accuracy:.2f}%")
            best_accuracy = test_accuracy
            best_epoch = epoch
            best_model_state_dict = model.state_dict()

            if config.modality == 'multimodal':
                best_path_with_accuracy = config.best_model_path.split('.pth')[0] + f'_epoch_{epoch+1}_a-acc_{test_accuracy_a:.2f}_v-acc_{test_accuracy_v:.2f}_acc_{test_accuracy:.2f}' + '.pth'
            else:
                best_path_with_accuracy = config.best_model_path.split('.pth')[0] + f'_epoch_{epoch+1}_acc_{test_accuracy:.2f}' + '.pth'
            torch.save(best_model_state_dict, best_path_with_accuracy)
            print(f"Best model found at epoch {best_epoch+1}, saved to {best_path_with_accuracy} with accuracy {best_accuracy:.2f}%")
        
        print(f"Best Accuracy: {best_accuracy:.2f}%")
        

        scheduler.step()

        
    print(f"Best model found at epoch {best_epoch+1}, saved to {best_path_with_accuracy} with accuracy {best_accuracy:.2f}%")    
                
    print("\n\n Evaluation on best TEACHER model is beginning...\n\n")
    model.load_state_dict(torch.load(best_path_with_accuracy))
    
    if config.modality == 'multimodal':
        train_loss_a, train_loss_v, train_loss, train_accuracy_a, train_accuracy_v, train_accuracy = evaluate(model, device, train_loader, criterion)
        test_loss_a, test_loss_v, test_loss, test_accuracy_a, test_accuracy_v, test_accuracy = evaluate(model, device, test_loader, criterion)

        print(f'Train Loss\t: {train_loss:.4f}, Train Acc\t : {train_accuracy:.2f}%, \t--\t Test Loss\t: {test_loss:.4f}, Test Acc\t: {test_accuracy:.2f}%, \n'
              f'Audio Train Loss: {train_loss_a:.4f}, Audio Train Acc: {train_accuracy_a:.2f}%, \t--\t Audio Test Loss: {test_loss_a:.4f}, Audio Test Acc: {test_accuracy_a:.2f}%, \n'
              f'Video Train Loss: {train_loss_v:.4f}, Video Train Acc: {train_accuracy_v:.2f}%, \t--\t Video Test Loss: {test_loss_v:.4f}, Video Test Acc: {test_accuracy_v:.2f}%\n')
    else:
        train_loss, train_accuracy = evaluate(model, device, train_loader, criterion)
        test_loss, test_accuracy = evaluate(model, device, test_loader, criterion)

        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, '
              f'Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.2f}%')

    writer.close()  # Close the writer after training

if __name__ == "__main__":
    main()
