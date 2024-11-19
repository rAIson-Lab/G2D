import config
import torch
import torch.nn as nn
import numpy as np
import random
import os


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
        

def unimodal_outputs(model, af, vf):
    fusion_module = model.fusion_module

    if config.fusion_method == 'sum':
        # SumFusion: Linear transformations on each modality separately, then add
        out_a = (torch.mm(af, torch.transpose(fusion_module.fc_x.weight, 0, 1)) +
                 fusion_module.fc_x.bias / 2)
        out_v = (torch.mm(vf, torch.transpose(fusion_module.fc_y.weight, 0, 1)) +
                 fusion_module.fc_y.bias / 2)

    elif config.fusion_method == 'late':
        # LateFusion: Similar to SumFusion but with averaging logits
        out_a = (torch.mm(af, torch.transpose(fusion_module.fc_x.weight, 0, 1)) +
                 fusion_module.fc_x.bias)
        out_v = (torch.mm(vf, torch.transpose(fusion_module.fc_y.weight, 0, 1)) +
                 fusion_module.fc_y.bias)

    elif config.fusion_method == 'concat':
        # ConcatFusion: Split weights in fc_out for each modality
        weight_size = fusion_module.fc_out.weight.size(1)
        out_a = (torch.mm(af, torch.transpose(fusion_module.fc_out.weight[:, :weight_size // 2], 0, 1))
                 + fusion_module.fc_out.bias / 2)
        out_v = (torch.mm(vf, torch.transpose(fusion_module.fc_out.weight[:, weight_size // 2:], 0, 1))
                 + fusion_module.fc_out.bias / 2)

    elif config.fusion_method == 'film':
        # FiLM: Use conditioning with gamma and beta from fc for each modality
        gamma_a, beta_a = torch.split(fusion_module.fc(af), fusion_module.dim, 1)
        out_a = gamma_a * af + beta_a
        out_a = torch.mm(out_a, torch.transpose(fusion_module.fc_out.weight, 0, 1)) + fusion_module.fc_out.bias / 2

        gamma_v, beta_v = torch.split(fusion_module.fc(vf), fusion_module.dim, 1)
        out_v = gamma_v * vf + beta_v
        out_v = torch.mm(out_v, torch.transpose(fusion_module.fc_out.weight, 0, 1)) + fusion_module.fc_out.bias / 2
    
    elif config.fusion_method == 'gated':
        out_a = torch.mm(af, torch.transpose(fusion_module.fc_out.weight, 0, 1)) + fusion_module.fc_out.bias / 2
        out_v = torch.mm(vf, torch.transpose(fusion_module.fc_out.weight, 0, 1)) + fusion_module.fc_out.bias / 2

    return out_a, out_v



def evaluate(model, device, test_loader, criterion):
    """
    Evaluate the model using the provided data loader.
    Handles different modalities based on the configuration.
    """
    model.eval()
    
    total_loss = 0
    total_loss_a = 0
    total_loss_v = 0
    
    total_correct = 0
    total_correct_a = 0
    total_correct_v = 0
    
    total_samples = 0
    
    with torch.no_grad():
        for data in test_loader:
            audio, video, labels = data
            audio, video, labels = audio.to(device), video.to(device), labels.to(device)
            if config.dataset == 'AVMNIST':
                audio = audio.float()
            else:
                audio = audio.unsqueeze(1).float()
            video = video.float()
            
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

    avg_loss = total_loss / len(test_loader)
    accuracy = 100 * (total_correct / total_samples)
    
    if config.modality == 'multimodal':
        avg_loss_a = total_loss_a / len(test_loader)
        avg_loss_v = total_loss_v / len(test_loader)
        accuracy_a = 100 * (total_correct_a / total_samples)
        accuracy_v = 100 * (total_correct_v / total_samples)
        return avg_loss_a, avg_loss_v, avg_loss, accuracy_a, accuracy_v, accuracy
    
    return avg_loss, accuracy


def calculate_accuracy(outputs, labels):
    outputs = nn.Softmax(dim=1)(outputs)
    _, preds = torch.max(outputs.data, 1)
    return (preds == labels).sum().item()



def update_model_with_OGM_GE(out_a, out_v, labels, model, epoch):
    'OGM_GE implementation from the CVPR 2022 Paper: "Balanced Multimodal Learning via On-the-fly Gradient Modulation"'
    
    softmax = nn.Softmax(dim=1)
    relu = nn.ReLU(inplace=True)
    tanh = nn.Tanh()
    
    # Convert labels to long for proper indexing
    labels = labels.long()
    
    # Modulation starts here !
    score_v = sum([softmax(out_v)[i][labels[i]] for i in range(out_v.size(0))])
    score_a = sum([softmax(out_a)[i][labels[i]] for i in range(out_a.size(0))])

    ratio_v = score_v / score_a
    ratio_a = 1 / ratio_v
    
    """
    Below is the Eq.(10) in our CVPR paper:
            1 - tanh(alpha * rho_t_u), if rho_t_u > 1
    k_t_u =
            1,                         else
    coeff_u is k_t_u, where t means iteration steps and u is modality indicator, either a or v.
    """
    
    if ratio_v > 1:
        coeff_v = 1 - tanh(config.alpha * relu(ratio_v))
        coeff_a = 1
    else:
        coeff_a = 1 - tanh(config.alpha * relu(ratio_a))
        coeff_v = 1
     
    if config.modulation_starts <= epoch <= config.modulation_ends:
        for name, parms in model.named_parameters():
            layer = str(name).split('.')[0]
            # print(f"layer: {layer}")

            if 'audio' in layer and len(parms.grad.size()) == 4:
                if config.modulation == 'OGM-GE':
                    # print("Doing OGM-GE Modulation:")  # For debugging
                    parms.grad = parms.grad * coeff_a + \
                                    torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-8)
                elif config.modulation == 'OGM':
                    # print("Doing OGM Modulation:")  # For debugging
                    parms.grad *= coeff_a

            if 'visual' in layer and len(parms.grad.size()) == 4:
                if config.modulation == 'OGM-GE':  
                    # print("Doing OGM-GE Modulation:")  # For debugging
                    parms.grad = parms.grad * coeff_v + \
                                    torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-8)
                elif config.modulation == 'OGM':
                    # print("Doing OGM Modulation:")  # For debugging
                    parms.grad *= coeff_v
    else:
        # print("Not doing Modulation.")
        pass  
    
    return ratio_v, coeff_a, coeff_v



def modulate_gradients_based_on_scores_from_two_modalities(teacher_out_a, teacher_out_v, out_a, out_v, labels, modulation_type, model, epoch):
    """
    Modulates distilled gradients from one modality
    
    Args:
        teacher_out_a: output of audio teacher
        teacher_out_v: output of visual teacher
        out_a: output of audio modality of student
        out_v: output of visual modality of student
        labels: labels 
        modulation_type: 'amplify_dominant', 'amplify_inferior', 'supress_dominant', 'supress_inferior'
        modality: 'audio' or 'visual'
        model: model
        epoch: current epoch
    
    
    Returns:
        ratio_v: ratio of score_v/score_a
        coeff_a: coefficient of score_a
        coeff_v: coefficient of score_v
    
    """
    
    softmax = nn.Softmax(dim=1)
    
    #print outs
    # print(f"teacher_out_a: {teacher_out_a}")
    # print(f"teacher_out_v: {teacher_out_v}")
    # print(f"out_a: {out_a}")
    # print(f"out_v: {out_v}")
    # print(f"labels: {labels}")
    
    # Modulation starts here !   
    score_a = sum([softmax(out_a)[i][labels[i]] for i in range(out_a.size(0))])
    score_v = sum([softmax(out_v)[i][labels[i]] for i in range(out_v.size(0))])
    score_teacher_a = sum([softmax(teacher_out_a)[i][labels[i]] for i in range(teacher_out_a.size(0))])
    score_teacher_v = sum([softmax(teacher_out_v)[i][labels[i]] for i in range(teacher_out_v.size(0))])
    
    coeff_a = 1.0
    coeff_v = 1.0
    
    if score_teacher_a > score_teacher_v:

                    
        if modulation_type == 'G2D':
            # print("Doing G2D Modulation:")
            coeff_a = 0
            coeff_v = 1
    
    elif score_teacher_a < score_teacher_v:

        if modulation_type == 'G2D':
            # print("Doing G2D Modulation:")
            coeff_a = 1
            coeff_v = 0
    
    else:
        raise ValueError(f"modulation_type should be G2D but got {modulation_type}")
    
        
    # save scores, ratios and coefficients in a csv file
    with open(os.path.join(config.scores_path, config.filename), 'a') as f:
        f.write(f"{epoch}, {coeff_a}, {coeff_v}, {score_teacher_a}, {score_teacher_v}, {score_a}, {score_v}\n") 
    
     
    if config.modulation_starts <= epoch <= config.modulation_ends:
        for name, parms in model.named_parameters():                       
            layer = str(name).split('.')[0]  
            # print(f"layer: {layer}")
            
            if 'audio' in layer and len(parms.data.size()) == 4:                 
                parms.grad = (parms.grad * coeff_a)
                    
            if 'visual' in layer and len(parms.data.size()) == 4: 
                parms.grad = (parms.grad * coeff_v)
                        
    else:
        # print("Not doing Modulation.")  
        pass                
    
    # print (f"coeff_a: {coeff_a}, coeff_v: {coeff_v}")                
    
    return 0, coeff_a, coeff_v