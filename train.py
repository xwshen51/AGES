import sys
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import TensorDataset, DataLoader
import argparse
import matplotlib.pyplot as plt

from bgm import *
from sagan import *
from config import *
import os
import random
import utils


def main():
    global args
    args = get_config()
    args.commond = 'python ' + ' '.join(sys.argv)

    # Create saving directory
    if args.unigen:
        save_dir = './results_unigen/{0}/G{1}_glr{2}_dlr{3}_dstep{4}_zdim{5}_{6}/'.format(
            args.dataset, args.dec_dist, str(args.lr), str(args.lr_d), str(args.d_steps_per_iter),
            str(args.latent_dim), args.div)
    else:
        save_dir = './results/{0}/E{1}_G{2}_glr{3}_dlr{4}_gstep{5}_dstep{6}_zdim{7}_{8}/'.format(
            args.dataset, args.enc_dist, args.dec_dist, str(args.lr), str(args.lr_d),
            str(args.g_steps_per_iter), str(args.d_steps_per_iter), str(args.latent_dim), args.div)

    utils.make_folder(save_dir)
    utils.write_config_to_file(args, save_dir)

    global device
    device = torch.device('cuda')

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Load datasets
    train_loader, test_loader = utils.make_dataloader(args)

    num_samples = len(train_loader.dataset)
    global num_iter_per_epoch
    num_iter_per_epoch = num_samples // args.batch_size

    # Losses file
    log_file_name = os.path.join(save_dir, 'log.txt')
    global log_file
    if args.resume:
        log_file = open(log_file_name, "at")
    else:
        log_file = open(log_file_name, "wt")


    # Build model
    if args.unigen:
        if args.dataset == 'mnist_stack':
            model = DCDecoder(args.latent_dim, 64, args.image_size, 3, args.dec_dist)
            discriminator = DCDiscriminator(args.d_conv_dim, args.image_size)
        else:
            model = Generator(args.latent_dim, args.g_conv_dim, args.image_size)
            discriminator = Discriminator(args.d_conv_dim, args.image_size)
        encoder_optimizer = None
        decoder_optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
        D_optimizer = optim.Adam(discriminator.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2))

    else:
        if args.dataset == 'mog':
            model = ToyAE(data_dim=2, latent_dim=args.latent_dim, enc_hidden_dim=500, dec_hidden_dim=500,
                          enc_dist=args.enc_dist, dec_dist=args.dec_dist)
            discriminator = DiscriminatorMLP(data_dim=2, latent_dim=args.latent_dim, hidden_dim_x=400,
                                             hidden_dim_z=400, hidden_dim=400)
        elif args.dataset in ['mnist', 'mnist_stack']:
            image_channel = 3 if args.dataset == 'mnist_stack' else 1
            tanh = args.prior == 'uniform' and args.enc_dist == 'deterministic'
            model = DCAE(args.latent_dim, 64, args.image_size, image_channel, args.enc_dist, args.dec_dist, tanh)
            discriminator = DCJointDiscriminator(args.latent_dim, 64, args.image_size, image_channel, args.dis_fc_size)
        else:
            model = BGM(args.latent_dim, args.g_conv_dim, args.image_size, 3,
                        args.enc_dist, args.enc_arch, args.enc_fc_size, args.enc_noise_dim, args.dec_dist)
            discriminator = BigJointDiscriminator(args.latent_dim, args.d_conv_dim, args.image_size, args.dis_fc_size)
        encoder_optimizer = optim.Adam(model.encoder.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
        decoder_optimizer = optim.Adam(model.decoder.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
        D_optimizer = optim.Adam(discriminator.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2))

    # Load model from checkpoint
    if args.resume:
        ckpt_dir = args.ckpt_dir if args.ckpt_dir != '' else save_dir + 'model' + str(args.start_epoch - 1) + '.sav'
        checkpoint = torch.load(ckpt_dir)
        model.load_state_dict(checkpoint['model'])
        discriminator.load_state_dict(checkpoint['discriminator'])
        del checkpoint

    model = nn.DataParallel(model.to(device))
    discriminator = nn.DataParallel(discriminator.to(device))


    # Fixed noise from prior p_z for generating from G
    global fixed_noise
    if args.prior == 'gaussian':
        fixed_noise = torch.randn(args.save_n_samples, args.latent_dim, device=device)
    else:
        fixed_noise = torch.rand(args.save_n_samples, args.latent_dim, device=device) * 2 - 1


    # Train
    for i in range(args.start_epoch, args.start_epoch + args.n_epochs):
        train_age(i, model, discriminator, encoder_optimizer, decoder_optimizer, D_optimizer, train_loader,
                  args.print_every, save_dir, args.sample_every, test_loader)
        if i % args.save_model_every == 0:
            torch.save({'model': model.module.state_dict(), 'discriminator': discriminator.module.state_dict()},
                       save_dir + 'model' + str(i) + '.sav')


# Training functions
def train_age(epoch, model, discriminator, encoder_optimizer, decoder_optimizer, D_optimizer, train_loader, print_every,
              save_dir, sample_every, test_loader=None):
    '''
    Training the bidirectional generative model using AGE.
    :param epoch: training epoch
    :param model: encoder E(x) and generator G(z)
    :param discriminator: discriminator D(x,z)
    :param encoder_optimizer: optimizer for encoder
    :param decoder_optimizer: optimizer for generator
    :param D_optimizer: optimizer for discriminator
    :param train_loader: training data loader
    :param print_every: print losses every print_every iterations
    :param save_dir: directory for saving sampled images and model
    :param sample_every: test (sample images) every sample_every iterations
    :param test_loader: test data loader (used on MoG)
    '''
    model.train()
    discriminator.train()

    for batch_idx, (x) in enumerate(train_loader):
        if not args.dataset in ['celeba1', 'celeba-1']:
            x = x[0]
        x = x.to(device)

        # ================== TRAIN DISCRIMINATOR ================== #
        for _ in range(args.d_steps_per_iter):
            discriminator.zero_grad()

            # Sample z from prior p_z
            if args.prior == 'gaussian':
                z = torch.randn(x.size(0), args.latent_dim, device=x.device)
            else:
                z = torch.rand(x.size(0), args.latent_dim, device=x.device) * 2 - 1

            # Get inferred latent z = E(x) and generated image x = G(z)
            if args.unigen:
                x_fake = model(z)
            else:
                z_fake, x_fake = model(x, z)

            # Compute D loss
            if args.unigen:
                # Real data score
                encoder_score = discriminator(x)
                # Fake data score
                decoder_score = discriminator(x_fake.detach())
            else:
                encoder_score = discriminator(x, z_fake.detach())
                decoder_score = discriminator(x_fake.detach(), z)
                del z_fake
            del x_fake

            loss_d = F.softplus(decoder_score).mean() + F.softplus(-encoder_score).mean()
            loss_d.backward()
            D_optimizer.step()

        for _ in range(args.g_steps_per_iter):
            if args.prior == 'gaussian':
                z = torch.randn(x.size(0), args.latent_dim, device=x.device)
            else:
                z = torch.rand(x.size(0), args.latent_dim, device=x.device) * 2 - 1
            if args.unigen:
                x_fake = model(z)
            else:
                z_fake, x_fake = model(x, z)

            # ================== TRAIN ENCODER ================== #
            if not args.unigen:
                model.zero_grad()
                encoder_score = discriminator(x, z_fake)
                del z_fake

                # Clip the scaling factor
                r_encoder = torch.exp(encoder_score.detach())
                if args.clip:
                    upper = 1 / args.scale_lower if args.scale_upper is None else args.scale_upper
                    r_encoder = r_encoder.clamp(args.scale_lower, upper)

                if args.div == 'revkl':
                    s_encoder = 1 / r_encoder
                elif args.div == 'js':
                    s_encoder = 1 / (1 + r_encoder)
                elif args.div == 'hellinger':
                    s_encoder = 1 / (2 * torch.sqrt(r_encoder))
                else:
                    assert args.div in ['all', 'kl']
                    s_encoder = r_encoder.new_ones(r_encoder.shape)
                loss_encoder = (s_encoder * encoder_score).mean()

                loss_encoder.backward()
                encoder_optimizer.step()

            # ================== TRAIN GENERATOR ================== #
            model.zero_grad()

            if args.unigen:
                decoder_score = discriminator(x_fake)
            else:
                decoder_score = discriminator(x_fake, z)

            # Clip the scaling factor
            r_decoder = torch.exp(decoder_score.detach())
            if args.clip:
                upper = 1 / args.scale_lower if args.scale_upper is None else args.scale_upper
                r_decoder = r_decoder.clamp(args.scale_lower, upper)

            if args.div == 'kl':
                s_decoder = r_decoder
            elif args.div == 'js':
                s_decoder = r_decoder / (r_decoder + 1)
            elif args.div == 'hellinger':
                s_decoder = torch.sqrt(r_decoder) / 2
            else:
                assert args.div in ['all', 'revkl']
                s_decoder = r_decoder.new_ones(r_decoder.shape)
            loss_decoder = -(s_decoder * decoder_score).mean()

            loss_decoder.backward()
            decoder_optimizer.step()

        # Print out losses
        if batch_idx == 0 or (batch_idx + 1) % print_every == 0:
            if args.unigen:
                log = ('Train Epoch: {} ({:.0f}%)\tD loss: {:.4f}, Decoder loss: {:.4f}'.format(
                    epoch, 100. * batch_idx / len(train_loader),
                    loss_d.item(), loss_decoder.item()))
            else:
                log = ('Train Epoch: {} ({:.0f}%)\tD loss: {:.4f}, Encoder loss: {:.4f}, Decoder loss: {:.4f}'.format(
                    epoch, 100. * batch_idx / len(train_loader),
                    loss_d.item(), loss_encoder.item(), loss_decoder.item()))
            print(log)
            log_file.write(log + '\n')
            log_file.flush()

        # Sample images
        if (batch_idx + 1) % sample_every == 0:
            if args.dataset != 'mog':
                if epoch % args.sample_every_epoch == 0:
                    test(epoch, batch_idx + 1, model, x[:args.save_n_recons], save_dir)
            else:
                plot = epoch % args.sample_every_epoch == 0 and (batch_idx + 1) + sample_every > num_iter_per_epoch
                test_toy(epoch, batch_idx + 1, model, test_loader, fixed_noise, save_dir, plot)


def test(epoch, i, model, test_data, save_dir):
    model.eval()
    with torch.no_grad():
        x = test_data.to(device)

        # Reconstruction
        if not args.unigen:
            x_recon = model(x, recon=True)
            recons = utils.draw_recon(x.cpu(), x_recon.cpu())
            del x_recon
            save_image(recons, save_dir + 'recon_' + str(epoch) + '_' + str(i) + '.png', nrow=args.nrow,
                       normalize=True, scale_each=True)

        # Generation
        sample = model(z=fixed_noise).cpu()
        save_image(sample, save_dir + 'gen_' + str(epoch) + '_' + str(i) + '.png', normalize=True, scale_each=True)
        del sample
    model.train()


def test_toy(epoch, i, model, test_loader, fixed_z, save_dir, plot=False):
    model.eval()
    with torch.no_grad():
        x_samp = model(z=fixed_z).detach().cpu()
    x_test, y_test = next(iter(test_loader))
    x_test = x_test.to(device)
    with torch.no_grad():
        if args.dec_dist == 'gaussian':
            x_recon, x_mu, x_logvar, z_mu, z_logvar = model(x=x_test, recon=True)
        else:
            x_recon = model(x=x_test, recon=True)
        x_recon = x_recon.detach().cpu()

    if plot:
        plt.rcParams['figure.figsize'] = [10, 5]
        plt.subplot(121)
        plt.scatter(x_samp[:, 0], x_samp[:, 1], marker='o')
        plt.subplot(122)
        plt.scatter(x_recon[:, 0], x_recon[:, 1], marker='o', c=y_test)
        plt.savefig(save_dir + str(epoch) + '.png')
        plt.close()

    if args.dec_dist == 'gaussian':
        nll = gaussian_nll(x_mu, x_logvar, x_test)
        kl_d = kl_div(z_mu, z_logvar)
        elbo = kl_d + nll

        log = ('Test Epoch {}, Iter {}\t-ELBO: {:.4f}, KL: {:.4f}, NLL: {:.4f}'.format(
            epoch, i, elbo.item() / len(x_test), kl_d.item() / len(x_test), nll.item() / len(x_test)))
        print(log)
        log_file.write(log + '\n')
        log_file.flush()

    model.train()


if __name__ == '__main__':
    main()
