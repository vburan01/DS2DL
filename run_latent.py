import argparse
from UMAE_run import UMAE_run

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_train_pixels', type=int, required=True)
    parser.add_argument('--mask_ratio', type=float, required=True)
    parser.add_argument('--dataset', type=str, default='Botswana')
    args = parser.parse_args()
    
    UMAE_run(
        dataset=args.dataset,
        gpu_id='0',
        seed=0,
        batch_size=48,
        patches=7,
        band_patches=5,
        epoches=100,
        learning_rate=5e-4,
        mask_ratio=args.mask_ratio,
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
        num_train_pixels=args.num_train_pixels,
        output_dir='./outputs/',
        save_ckpt_freq=200,
        experiment_name='',
        extract_batch_size=256,
        mode='ViT',
        trained_model=''
    )

