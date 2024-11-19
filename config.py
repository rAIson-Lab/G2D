import os
import torch

DEBUG = False
# DEBUG = True

# Random seed for reproducibility
random_seed = 999

## Experiment Names
# Multimodal Student
exp_name = 'aT+aTF+vT+vTF-to-mS' # audio teacher logits, audio teacher feature, visual teacher logits, visual teacher feature to multimodal student
# exp_name = 'aTF+vTF-to-mS' # audio teacher feature, visual teacher feature to multimodal student
# Teacher Training
# exp_name = 'aT' # audio teacher
# exp_name = 'vT' # visual teacher
# exp_name = 'mT' # multimodal teacher

# Experiment Modulation Type
# modulation = 'Normal'
# modulation = 'OGM'
# modulation = 'OGM-GE'

## My Modulation Methods
modulation = 'G2D'    


# Dataset settings
dataset = 'CREMAD'
# dataset = 'VGGSound'
# dataset = 'AVMNIST'


# Training settings
if dataset == 'CREMAD' or dataset == 'VGGSound' or dataset == 'AVMNIST':
    batch_size = 16
    epochs = 300
    lr_decay_step = 200
    learning_rate = 0.001
    lr_decay_ratio = 0.1 # gets multiplied with the lr
    # optimiser = 'adam'
    optimiser = 'sgd'
    weight_decay = 1e-4
    # weight_decay = 0.0

    # OGM-GE params
    if modulation != 'Normal':
        # Controls the degree of modulation: 0.8 for CREMA-D and 0.1 for VGGSound for OGM-GE. For my methods, alpha is always 1.
        alpha = 1.0
        modulation_starts = 0 # Epochs at which to start modulation
        modulation_ends = 150 # Epochs at which to end modulation
        modulation_string = f"{modulation}_alpha_{alpha}_modS_{modulation_starts}_modE_{modulation_ends}"
    else:
        modulation_string = modulation
        
    modulation_string = f"{modulation_string}_lr_{learning_rate}_lr-dstep_{lr_decay_step}_lr-dratio_{lr_decay_ratio}_bt_{batch_size}_{optimiser}_wd_{weight_decay}"

# Set modality based on exp name
if exp_name == 'aT':
    modality = 'audio'
elif exp_name == 'vT':
    modality = 'visual'
elif (exp_name == 'mT') or "mS" in exp_name:
    modality = 'multimodal'
else:
    raise NotImplementedError(f"Incorrect experiment name {exp_name}")

if modality == 'multimodal':
    # fusion_method = 'concat' # Options: 'sum', 'concat', 'gated', 'film'
    # fusion_method = 'sum'
    # fusion_method = 'film'
    # fusion_method = 'gated'
    fusion_method = 'late' 
else:
    fusion_method = None

# Model settings
if exp_name in ('aT', 'vT', 'mT'):
    role = 'teacher'
else:
    role = 'student'


if DEBUG:
    epochs = 1
    

if dataset == 'CREMAD':
    train_path = '/home/USERNAME/Multi-modal-Imbalance/data/CREMAD/train.csv'  # Path to train data
    test_path = '/home/USERNAME/Multi-modal-Imbalance/data/CREMAD/test.csv'  # Path to test data
    visual_path = '/home/USERNAME/Multimodal-Datasets/CREMA-D/Image-01-FPS'  # Path to visual data (01-FPS for 1 fps, 02-FPS for 2 fps and so on)
    audio_path = '/home/USERNAME/Multimodal-Datasets/CREMA-D/AudioWAV' # Path to audio data
elif dataset == 'VGGSound':
    pass ## All paths defined in VGGSoundDataset.py
elif dataset == 'AVMNIST':
    data_root = '/home/USERNAME/Multimodal-Datasets/AV-MNIST/avmnist'  # Path to AV-MNIST dataset


# Saved Model settings
if role == 'student':
    ckpt_path = f'/home/USERNAME/G2D/checkpoints/{dataset}/student'  # Path to save student models
elif role == 'teacher':
    ckpt_path = f'/home/USERNAME/G2D/checkpoints/{dataset}/teacher'  # Path to save teacher models


# Loss function weights
if role == 'student':    
    ce_loss_weight = 1.0
    
    if 'aT' in exp_name:
        if dataset == 'CREMAD':
            audio_teacher_weights = "/home/USERNAME/MM-KD/checkpoints/CREMAD/teacher/best_model_s999_aT_CREMAD_teacher_epoch_148_acc_61.69.pth"
        elif dataset == 'AVMNIST':
            audio_teacher_weights = "/home/USERNAME/MM-KD/checkpoints/AVMNIST/teacher/best_model_s999_aT_AVMNIST_teacher_epoch_152_acc_42.74.pth"
        elif dataset == 'VGGSound':
            audio_teacher_weights = "/home/USERNAME/MM-KD/checkpoints/VGGSound/teacher/best_model_s999_aT_VGGSound_teacher_epoch_274_acc_43.39.pth"
        if 'aT+aTF' in exp_name:
            logit_audio_loss_weight = 1.0
            audio_feature_loss_weight = 1.0
            temp_audio = 1
        elif 'aTF' in exp_name:
            audio_feature_loss_weight = 1.0
            temp_audio = 1
        elif 'aT' in exp_name:
            logit_audio_loss_weight = 1.0
            temp_audio = 1
     
    if 'vT' in exp_name:
        if dataset == 'CREMAD':
            video_teacher_weights = "/home/USERNAME/MM-KD/checkpoints/CREMAD/teacher/best_model_s999_vT_CREMAD_teacher_epoch_298_acc_76.48.pth"
        elif dataset == 'AVMNIST':
            video_teacher_weights = "/home/USERNAME/MM-KD/checkpoints/AVMNIST/teacher/best_model_s999_vT_AVMNIST_teacher_epoch_91_acc_65.59.pth"
        elif dataset == 'VGGSound':
            video_teacher_weights = "/home/USERNAME/MM-KD/checkpoints/VGGSound/teacher/best_model_s999_vT_VGGSound_teacher_epoch_272_acc_32.32.pth"
        if 'vT+vTF' in exp_name:
            logit_video_loss_weight = 1.0
            video_feature_loss_weight = 1.0
            temp_video = 1
        elif 'vTF' in exp_name:
            video_feature_loss_weight = 1.0
            temp_video = 1
        elif 'vT' in exp_name:
            logit_video_loss_weight = 1.0
            temp_video = 1
            
    # Add ce_loss_weight to modulation_string
    if modulation == 'Normal' or modulation == 'OGM' or modulation == 'OGM-GE':
        pass
    else:
        # write scores.csv with titles
        filename = f'scores_{dataset}_{modulation}.csv'
        
        scores_path = f'/home/USERNAME/G2D/scores/{dataset}'
        if not os.path.exists(scores_path):
            os.makedirs(scores_path)

        
# Image and spectrogram dimensions
img_width = 224
img_height = 224
spectrogram_width = 512
spectrogram_height = 128
fps = 1
use_video_frames = 3 # For VGGSound dataset

# Only use GPU if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Adjust GPU settings if cuda is available
if device.type == 'cuda':
    if os.environ.get('CUDA_VISIBLE_DEVICES') is None:
        gpu_ids = list(range(torch.cuda.device_count()))
    else:
        gpu_ids = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    gpu_ids = list(range(torch.cuda.device_count()))
    device_count = len(gpu_ids)
    print("Training on GPU")
else:
    gpu_ids = None
    device_count = 1
    print("Using CPU")
    

if modality == 'multimodal':
    ckpt_string = f'best_model_s{random_seed}_{exp_name}_{modulation_string}_{fusion_method}_{dataset}_{role}'
    if DEBUG:
        ckpt_string = f'best_model_DEBUG_s{random_seed}_{exp_name}_{modulation_string}_{fusion_method}_{dataset}_{role}'
    best_model_path = os.path.join(ckpt_path, ckpt_string + '.pth')
else:
    ckpt_string = f'best_model_s{random_seed}_{exp_name}_{dataset}_{role}'
    if DEBUG:
        ckpt_string = f'best_model_DEBUG_s{random_seed}_{exp_name}_{dataset}_{role}'
    best_model_path = os.path.join(ckpt_path, ckpt_string + '.pth')
    

if not os.path.exists(ckpt_path):
    os.makedirs(ckpt_path)


if role == 'student':
    if 'aT' in exp_name and 'vT' in exp_name:
        temperature = f"ta{temp_audio}_tv{temp_video}"
    elif 'aT' in exp_name:
        temperature = f"ta{temp_audio}"
    elif 'vT' in exp_name:
        temperature = f"tv{temp_video}"


def get_hparams():
    """
    Returns a dictionary of all relevant hyperparameters for logging or tracking.
    All variables are accessed directly from the local scope as they are defined in config.py.
    """
    hparams = {
        'exp_name': exp_name,
        'modulation': modulation,
        'dataset': dataset,
        'role': role,
        'modality': modality,
        'batch_size': batch_size,
        'epochs': epochs,
        'learning_rate': learning_rate,
        'lr_decay_step': lr_decay_step,
        'lr_decay_ratio': lr_decay_ratio,
        'random_seed': random_seed,
        'fusion_method': fusion_method,
        'alpha': str(alpha) if 'alpha' in locals() else 'None',  # Only applicable if OGM-GE or modulation is set
        'modulation_starts': str(modulation_starts) if 'modulation_starts' in locals() else 'None',
        'modulation_ends': str(modulation_ends) if 'modulation_ends' in locals() else 'None',
        'device': str(device),  # Convert torch.device to string
        'gpu_ids': str(gpu_ids) if device.type == 'cuda' else 'N/A',  # Convert list to string
    }

    # Add teacher-specific or student-specific hyperparameters
    if role == 'student':
        hparams.update({
            'ce_loss_weight': ce_loss_weight,
            'logit_audio_loss_weight': str(logit_audio_loss_weight) if 'logit_audio_loss_weight' in locals() else 'None',
            'audio_feature_loss_weight': str(audio_feature_loss_weight) if 'audio_feature_loss_weight' in locals() else 'None',
            'logit_video_loss_weight': str(logit_video_loss_weight) if 'logit_video_loss_weight' in locals() else 'None',
            'video_feature_loss_weight': str(video_feature_loss_weight) if 'video_feature_loss_weight' in locals() else 'None',
            'temp_audio': str(temp_audio) if 'temp_audio' in locals() else 'None',
            'temp_video': str(temp_video) if 'temp_video' in locals() else 'None',
        })

    # Paths and model saving info
    hparams['best_model_path'] = str(best_model_path)  # Convert Path to string

    return hparams


def print_hparams():
    """
    Prints all hyperparameters in a clean, organized manner.
    """
    hparams = get_hparams()
    print("\nHyperparameters:")
    for key, value in hparams.items():
        print(f"{key}: {value}")
    print("\n")
        
# Print the hyperparameters
print_hparams()
