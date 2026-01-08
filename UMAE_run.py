import os
import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
from pretrain_models import PretrainVisionTransformer, VisionTransformerEncoder
from scipy.io import savemat, loadmat
import torch.backends.cudnn as cudnn
from collections import OrderedDict
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

class RandomMaskingGenerator:
    def __init__(self, number_patches, mask_ratio):
        self.number_patches = number_patches
        self.num_mask = int(mask_ratio * self.number_patches)

    def __repr__(self):
        repr_str = "Maks: total patches {}, mask patches {}".format(
            self.number_patches, self.num_mask
        )
        return repr_str

    def __call__(self):
        mask = np.hstack([
            np.zeros(self.number_patches - self.num_mask),
            np.ones(self.num_mask),
        ])
        np.random.shuffle(mask)
        return mask
#-------------------------------------------------------------------------------
class AverageMeter(object):
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0
    
    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt
#-------------------------------------------------------------------------------        
def mirror_hsi(height, width, band, input_normalize, patch=5):
    padding = patch // 2
    mirror_hsi = np.zeros((height + 2 * padding, width + 2 * padding, band), dtype=float)
    mirror_hsi[padding:(padding + height), padding:(padding + width), :] = input_normalize
    for i in range(padding):
        mirror_hsi[padding:(height + padding), i, :] = input_normalize[:, padding - i - 1, :]
    for i in range(padding):
        mirror_hsi[padding:(height + padding), width + padding + i, :] = input_normalize[:, width - 1 - i, :]
    for i in range(padding):
        mirror_hsi[i, :, :] = mirror_hsi[padding * 2 - i - 1, :, :]
    for i in range(padding):
        mirror_hsi[height + padding + i, :, :] = mirror_hsi[height + padding - 1 - i, :, :]
    return mirror_hsi
#-------------------------------------------------------------------------------
def gain_neighborhood_pixel(mirror_image, point, i, patch=5):
    x = point[i, 0]
    y = point[i, 1]
    temp_image = mirror_image[x:(x + patch), y:(y + patch), :]
    return temp_image
#-------------------------------------------------------------------------------
def gain_neighborhood_band(x_train, band, band_patch, patch=5):
    nn = band_patch // 2
    pp = (patch * patch) // 2
    x_train_reshape = x_train.reshape(x_train.shape[0], patch * patch, band)
    x_train_band = np.zeros((x_train.shape[0], patch * patch * band_patch, band), dtype=np.float32)
    x_train_band[:, nn * patch * patch:(nn + 1) * patch * patch, :] = x_train_reshape
    for i in range(nn):
        if pp > 0:
            x_train_band[:, i * patch * patch:(i + 1) * patch * patch, :i + 1] = x_train_reshape[:, :, band - i - 1:]
            x_train_band[:, i * patch * patch:(i + 1) * patch * patch, i + 1:] = x_train_reshape[:, :, :band - i - 1]
        else:
            x_train_band[:, i:(i + 1), :(nn - i)] = x_train_reshape[:, 0:1, (band - nn + i):]
            x_train_band[:, i:(i + 1), (nn - i):] = x_train_reshape[:, 0:1, :(band - nn + i)]
    for i in range(nn):
        if pp > 0:
            x_train_band[:, (nn + i + 1) * patch * patch:(nn + i + 2) * patch * patch, :band - i - 1] = x_train_reshape[:, :, i + 1:]
            x_train_band[:, (nn + i + 1) * patch * patch:(nn + i + 2) * patch * patch, band - i - 1:] = x_train_reshape[:, :, :i + 1]
        else:
            x_train_band[:, (nn + 1 + i):(nn + 2 + i), (band - i - 1):] = x_train_reshape[:, 0:1, :(i + 1)]
            x_train_band[:, (nn + 1 + i):(nn + 2 + i), :(band - i - 1)] = x_train_reshape[:, 0:1, (i + 1):]
    return x_train_band
#-------------------------------------------------------------------------------
def train_and_test_data(mirror_image, band, train_point, test_point, true_point, patch=5, band_patch=3, flag='train'):
    x_train = np.zeros((train_point.shape[0], patch, patch, band), dtype=np.float32)
    x_test = np.zeros((test_point.shape[0], patch, patch, band), dtype=np.float32)
    x_true = np.zeros((true_point.shape[0], patch, patch, band), dtype=np.float32)
    for i in range(train_point.shape[0]):
        x_train[i, :, :, :] = gain_neighborhood_pixel(mirror_image, train_point, i, patch)
    for j in range(test_point.shape[0]):
        x_test[j, :, :, :] = gain_neighborhood_pixel(mirror_image, test_point, j, patch)
    for k in range(true_point.shape[0]):
        x_true[k, :, :, :] = gain_neighborhood_pixel(mirror_image, true_point, k, patch)
    if flag == 'test':
        x_test_band = gain_neighborhood_band(x_test, band, band_patch, patch)
        x_true_band = gain_neighborhood_band(x_true, band, band_patch, patch)
        x_train_band = x_train
    else:
        x_train_band = gain_neighborhood_band(x_train, band, band_patch, patch)
        x_test_band = gain_neighborhood_band(x_test, band, band_patch, patch)
        x_true_band = x_true
    return x_train_band, x_test_band, x_true_band
#-------------------------------------------------------------------------------
def select_pixels_spectral_pca_fps(
    input_normalize,
    num_train_pixels,
    seed=0,
    n_pca=20
):
    height, width, bands = input_normalize.shape
    N = height * width
    pixels = input_normalize.reshape(-1, bands)
    n_pca = min(n_pca, bands)
    pca = PCA(n_components=n_pca, random_state=seed)
    X_pca = pca.fit_transform(pixels)
    rng = np.random.default_rng(seed)
    first_idx = rng.integers(0, N)
    selected = [first_idx]
    diff = X_pca - X_pca[first_idx]
    min_dist_sq = np.sum(diff * diff, axis=1)
    for _ in range(1, num_train_pixels):
        next_idx = int(np.argmax(min_dist_sq))
        selected.append(next_idx)
        diff = X_pca - X_pca[next_idx]
        dist_sq_new = np.sum(diff * diff, axis=1)
        min_dist_sq = np.minimum(min_dist_sq, dist_sq_new)
    selected = np.array(selected, dtype=int)
    sampled_pixel_positions = np.array(
        [np.unravel_index(idx, (height, width)) for idx in selected],
        dtype=int
    )
    return sampled_pixel_positions, selected
#-------------------------------------------------------------------------------
def UMAE_run( 
    dataset='KSC',
    gpu_id='0',
    seed=0,
    batch_size=48,
    patches=7,
    band_patches=5,
    epoches=100,
    learning_rate=5e-4,
    mask_ratio=0.5,
    latent_dim=48,
    decoder_dim=48,
    encoder_depth=6,
    decoder_depth=4,
    encoder_heads=6,
    decoder_heads=4,
    mlp_dim=192,
    gamma=0.9,
    weight_decay=0.0001,
    init_scale=0.001,
    dropout=0.1,
    emb_dropout=0.1,
    num_train_pixels=400,
    output_dir='./outputs/',
    save_ckpt_freq=200,
    experiment_name='',
    extract_batch_size=256,
    mode='ViT',
    trained_model='',
):

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    cudnn.benchmark = True
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    if dataset == 'Botswana':
        data = loadmat('./datasets/Botswana.mat')
    elif dataset == 'KSC':
        data = loadmat('./datasets/KSC.mat')
    else:
        raise ValueError("Unknown dataset")
    if dataset == 'Botswana':
        input = data['Botswana']
    elif dataset == 'KSC':
        input = data['KSC']
    else:
        input = data['input']

    input_normalize = np.zeros(input.shape)
    for i in range(input.shape[2]):
        input_max = np.max(input[:,:,i])
        input_min = np.min(input[:,:,i])
        input_normalize[:,:,i] = (input[:,:,i]-input_min)/(input_max-input_min)
    height, width, band = input.shape
    number_patches = band
    
    mirror_image = mirror_hsi(height, width, band, input_normalize, patch=patches)
    
    all_pixel_positions = np.array([[i, j] for i in range(height) for j in range(width)])
    num_train_pixels = min(num_train_pixels, all_pixel_positions.shape[0])
    np.random.seed(seed)

    sampled_pixel_positions, sample_indices = select_pixels_spectral_pca_fps(
        input_normalize=input_normalize,
        num_train_pixels=num_train_pixels,
        seed=seed,
        n_pca=20
    )

    train_pixel_ids = sample_indices.copy()
    x_train_band, _, _ = train_and_test_data(
        mirror_image, band, 
        sampled_pixel_positions,
        sampled_pixel_positions,
        sampled_pixel_positions,
        patch=patches, 
        band_patch=band_patches, 
        flag='train'
    )

    masked_positional_generator = RandomMaskingGenerator(number_patches, mask_ratio)
    x_train = torch.from_numpy(np.transpose(x_train_band, (0, 2, 1))).type(torch.FloatTensor)
    bool_masked_pos_t = torch.zeros(x_train.shape[0], number_patches)
    for b in range(x_train.shape[0]):
        bool_masked_pos_t[b, :] = torch.from_numpy(masked_positional_generator())
    bool_masked_pos_t = bool_masked_pos_t > 0
    
    pixel_ids_tensor = torch.from_numpy(train_pixel_ids.astype(np.int64))
    Label_train = Data.TensorDataset(x_train, bool_masked_pos_t, pixel_ids_tensor)
    label_train_loader = Data.DataLoader(Label_train, batch_size=batch_size, shuffle=True)
    size_patches = band_patches * patches ** 2
    
    model = PretrainVisionTransformer(
        image_size = patches,
        near_band = band_patches,
        num_patches = number_patches,
        encoder_num_classes=0,
        encoder_dim=latent_dim,
        encoder_depth=encoder_depth,
        encoder_heads=encoder_heads,
        encoder_dim_head=latent_dim // encoder_heads,
        encoder_mode = mode,
        decoder_num_classes=size_patches,
        decoder_dim=decoder_dim,
        decoder_depth=decoder_depth,
        decoder_heads=decoder_heads,
        decoder_dim_head=decoder_dim // decoder_heads,
        decoder_mode=mode,
        mlp_dim = mlp_dim,
        dropout = dropout,
        emb_dropout = emb_dropout,
        mask_ratio = mask_ratio)
    model = model.cuda()
    
    criterion = nn.MSELoss().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=epoches//10, gamma=gamma, verbose=True)

    import time
    print("Starting training")
    training_start_time = time.time()
    for epoch in range(epoches):
        model.train()
        objs = AverageMeter()
        for batch_idx, (batch_data, batch_mask, batch_ids) in enumerate(label_train_loader):
            B, band_dim, size_patches_dim = batch_data.shape
            batch_data = batch_data.cuda()
            batch_mask = batch_mask.cuda()
            batch_ids_np = batch_ids.numpy().astype(np.int64)
            batch_target = []
            for i in range(B):
                masked_bands = batch_data[i][batch_mask[i]]
                batch_target.append(masked_bands)
            batch_target = torch.stack(batch_target, dim=0)
            optimizer.zero_grad()
            batch_pred = model(batch_data, batch_mask)
            loss = criterion(batch_pred, batch_target)
            loss.backward()
            optimizer.step()
            n = batch_data.shape[0]
            objs.update(loss.data, n)
        scheduler.step()
        print("Epoch: {:03d} train_loss: {:.8f}".format(epoch+1, objs.avg))
        if (epoch + 1) % save_ckpt_freq == 0 or epoch + 1 == epoches:
            checkpoint = {
                'model': model.state_dict(),
                'epoch': epoch+1,
                'hyperparameters': {
                    'dataset': dataset,
                    'batch_size': batch_size,
                    'patches': patches,
                    'band_patches': band_patches,
                    'epoches': epoches,
                    'learning_rate': learning_rate,
                    'mask_ratio': mask_ratio,
                    'latent_dim': latent_dim,
                    'decoder_dim': decoder_dim,
                    'encoder_depth': encoder_depth,
                    'decoder_depth': decoder_depth,
                    'encoder_heads': encoder_heads,
                    'decoder_heads': decoder_heads,
                    'mlp_dim': mlp_dim,
                    'num_train_pixels': num_train_pixels,
                    'experiment_name': experiment_name
                }
            }
            os.makedirs(output_dir, exist_ok=True)
            checkpoint_filename = os.path.join(output_dir, f'{experiment_name}_checkpoint-{epoch+1}.pth')
            torch.save(checkpoint, checkpoint_filename)
            print(f"Saved checkpoint at epoch {epoch+1}")
    training_end_time = time.time()
    total_training_time = training_end_time - training_start_time
    print(f"Training completed in {total_training_time:.2f} seconds ({total_training_time/60:.2f} minutes)")

    print("Extraction:")
    extraction_start_time = time.time()
    checkpoint_path = os.path.join(output_dir, f'{experiment_name}_checkpoint-{epoches}.pth')
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval()
    dataset_file_map = {
        'Botswana': 'Botswana.mat',
        'KSC': 'KSC.mat'
    }
    mat_file = dataset_file_map.get(dataset)
    if mat_file is None:
        raise ValueError(f"Unknown dataset {dataset} for extraction phase.")
    data = loadmat(f'./datasets/{mat_file}')
    if  dataset == 'Botswana':
        input_data = data['Botswana']
    elif dataset == 'KSC':
        input_data = data['KSC']
    else:
        input_data = data['input']
    input_normalize = np.zeros(input_data.shape)
    for i in range(input_data.shape[2]):
        input_max = np.max(input_data[:, :, i])
        input_min = np.min(input_data[:, :, i])
        input_normalize[:, :, i] = (input_data[:, :, i] - input_min) / (input_max - input_min)
    height, width, band = input_data.shape
    mirror_image = mirror_hsi(height, width, band, input_normalize, patch=patches)
    padding = patches // 2
    all_positions = []
    for i in range(height):
        for j in range(width):
            all_positions.append([i + padding, j + padding])
    all_positions = np.array(all_positions)
    total_pixels = len(all_positions)
    print(f" Processing {total_pixels} pixels in batches of {extract_batch_size}")
    all_latent_features = []
    for start_idx in range(0, total_pixels, extract_batch_size):
        end_idx = min(start_idx + extract_batch_size, total_pixels)
        batch_positions = all_positions[start_idx:end_idx]
        batch_patches = np.zeros((len(batch_positions), patches, patches, band), dtype=np.float32)
        for i, pos in enumerate(batch_positions):
            x, y = pos[0], pos[1]
            patch = mirror_image[x-padding:(x+padding+1), y-padding:(y+padding+1), :]
            batch_patches[i] = patch
        batch_data = gain_neighborhood_band(batch_patches, band, band_patches, patches)
        batch_tensor = torch.from_numpy(batch_data.transpose(0, 2, 1)).type(torch.FloatTensor).cuda()
        masking_pos = torch.zeros(batch_tensor.shape[0], band, dtype=torch.bool).cuda()
        with torch.no_grad():
            latent_vectors = model.encoder(batch_tensor, masking_pos=masking_pos)
            if latent_vectors.dim() == 3:
                latent_vectors = latent_vectors.mean(dim=1)
            all_latent_features.append(latent_vectors.cpu().numpy())
    all_latent_features = np.concatenate(all_latent_features, axis=0)
    latent_spatial = all_latent_features.reshape(height, width, -1)
    extraction_end_time = time.time()
    total_extraction_time = extraction_end_time - extraction_start_time
    print(f"Extraction completed in {total_extraction_time:.2f} seconds ({total_extraction_time/60:.2f} minutes)")
    
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, f"{dataset}_FPS{num_train_pixels}_MR{mask_ratio}.mat")
    savemat(output_filename, {
        'latent_features': all_latent_features.astype(np.float32),
        'latent_spatial': latent_spatial.astype(np.float32),
        'height': int(height),
        'width': int(width),
        'feature_dim': int(all_latent_features.shape[1]),
        'total_pixels': int(all_latent_features.shape[0]),
        'checkpoint_name': experiment_name,
        'model_params': {
            'patches': int(patches),
            'band_patches': int(band_patches),
            'dataset': dataset,
        },
        'all_args': {
            'dataset': dataset,
            'gpu_id': gpu_id,
            'seed': seed,
            'batch_size': batch_size,
            'patches': patches,
            'band_patches': band_patches,
            'epoches': epoches,
            'learning_rate': learning_rate,
            'mask_ratio': mask_ratio,
            'latent_dim': latent_dim,
            'decoder_dim': decoder_dim,
            'encoder_depth': encoder_depth,
            'decoder_depth': decoder_depth,
            'encoder_heads': encoder_heads,
            'decoder_heads': decoder_heads,
            'mlp_dim': mlp_dim,
            'gamma': gamma,
            'weight_decay': weight_decay,
            'init_scale': init_scale,
            'output_dir': output_dir,
            'save_ckpt_freq': save_ckpt_freq,
            'experiment_name': experiment_name,
            'extract_batch_size': extract_batch_size,
            'dropout': dropout,
            'emb_dropout': emb_dropout,
            'num_train_pixels': num_train_pixels,
            'trained_model': trained_model
        },
        'training_time': total_training_time,
        'extraction_time': total_extraction_time,
        'total_runtime': total_training_time + total_extraction_time,
        'description': f'Latent features from UMAE {experiment_name}'
    })
    print(f"Latent features saved to {output_filename}")
    return all_latent_features, latent_spatial
