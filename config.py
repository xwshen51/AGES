import argparse


def get_config():

    parser = argparse.ArgumentParser(description='Training (bidirectional) generative models using AGE')

    # Data settings
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--dataset', type=str, default='celeba',
                        choices=["celeba", "cifar", "imagenet", 'mnist', 'mnist_stack', 'mog'])
    parser.add_argument('--data_dir', type=str, default='~/data', help='data directory')
    # Settings of mixture of Gaussians
    parser.add_argument('--mog_imbalance', action='store_true', help='imbalanced mixture of Gaussians')
    parser.add_argument('--mog_base', type=int, default=3, help='square root of number of classes in MoG')
    parser.add_argument('--mog_std', type=float, default=0.3, help='standard deviation of each component of MoG')

    # Training settings
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate of encoder and generator')
    parser.add_argument('--lr_d', type=float, default=0.0001, help='learning rate of discriminator')
    parser.add_argument('--beta1', type=float, default=0.0)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--d_steps_per_iter', type=int, default=1, help='how many D updates per iteration')
    parser.add_argument('--g_steps_per_iter', type=int, default=1, help='how many G updates per iteration')
    parser.add_argument('--n_epochs', type=int, default=1000)
    parser.add_argument('--start_epoch', type=int, default=1)
    parser.add_argument('--seed', type=int, default=51)

    # Model hyper-parameters
    parser.add_argument('--latent_dim', type=int, default=100)
    parser.add_argument('--unigen', action='store_true', help='unidirectional generative model')
    parser.add_argument('--prior', type=str, default='gaussian', choices=['gaussian', 'uniform'], help='latent prior p_z')
    parser.add_argument('--div', type=str, default='all', choices=['all', 'kl', 'js', 'hellinger', 'revkl'],
                        help='objective divergence')
    parser.add_argument('--clip', action='store_true', help='clip the scaling factors')
    parser.add_argument('--scale_lower', type=float, default=0.5, help='lower bound of the scaling factor')
    parser.add_argument('--scale_upper', type=float, default=None, help='upper bound of the scaling factor')

    # Encoder settings
    parser.add_argument('--enc_arch', type=str, default='resnet', choices=['resnet', 'resnet18', 'dcgan'],
                        help='encoder architecture')
    parser.add_argument('--enc_dist', type=str, default='gaussian', choices=['deterministic', 'gaussian', 'implicit'],
                        help='encoder distribution')
    parser.add_argument('--enc_fc_size', type=int, default=2048, help='number of nodes in fc layer of resnet')
    parser.add_argument('--enc_noise_dim', type=int, default=128, help='')
    # Generator settings
    parser.add_argument('--dec_arch', type=str, default='sagan', choices=['sagan', 'dcgan'],
                        help='generator/decoder architecture')
    parser.add_argument('--dec_dist', type=str, default='deterministic', choices=['deterministic', 'gaussian', 'implicit'],
                        help='generator distribution')
    parser.add_argument('--g_conv_dim', type=int, default=32, help='base number of channels in encoder and generator')
    # Discriminator settings
    parser.add_argument('--dis_fc_size', type=int, default=1024, help='number of nodes in fc layer of joint discriminator')
    parser.add_argument('--d_conv_dim', type=int, default=32, help='base number of channels in discriminator')

    # Pretrained model
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--ckpt_dir', type=str, default='')

    # Output and save
    parser.add_argument('--print_every', type=int, default=100)
    parser.add_argument('--sample_every', type=int, default=600)
    parser.add_argument('--sample_every_epoch', type=int, default=10)
    parser.add_argument('--save_model_every', type=int, default=5)
    parser.add_argument('--save_n_samples', type=int, default=64)
    parser.add_argument('--save_n_recons', type=int, default=32)
    parser.add_argument('--nrow', type=int, default=8)

    args = parser.parse_args()

    return args