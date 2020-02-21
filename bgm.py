from sagan import *
import torchvision.models as models
from resnet import *
import torch.nn.init as init


class ResEncoder(nn.Module):
    r'''ResNet Encoder

    Args:
        latent_dim: latent dimension
        arch: network architecture. Choices: resnet - resnet50, resnet18
        dist: encoder distribution. Choices: deterministic, gaussian, implicit
        fc_size: number of nodes in each fc layer
        noise_dim: dimension of input noise when an implicit encoder is used
    '''
    def __init__(self, latent_dim=64, arch='resnet', dist='gaussian', fc_size=2048, noise_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.dist = dist
        self.noise_dim = noise_dim

        in_channels = noise_dim + 3 if dist == 'implicit' else 3
        out_dim = latent_dim * 2 if dist == 'gaussian' else latent_dim
        if arch == 'resnet':
            self.encoder = resnet50(pretrained=False, in_channels=in_channels, fc_size=fc_size, out_dim=out_dim)
        else:
            assert arch == 'resnet18'
            self.encoder = resnet18(pretrained=False, in_channels=in_channels, fc_size=fc_size, out_dim=out_dim)

    def forward(self, x, avepool=False):
        '''
        :param x: input image
        :param avepool: whether to return the average pooling feature (used for downstream tasks)
        :return:
        '''
        if self.dist == 'implicit':
            # Concatenate noise with the input image x
            noise = x.new(x.size(0), self.noise_dim, 1, 1).normal_(0, 1)
            noise = noise.expand(x.size(0), self.noise_dim, x.size(2), x.size(3))
            x = torch.cat([x, noise], dim=1)
        z, ap = self.encoder(x)
        if avepool:
            return ap
        if self.dist == 'gaussian':
            return z.chunk(2, dim=1)
        else:
            return z


class BigDecoder(nn.Module):
    r'''Big generator based on SAGAN

    Args:
        latent_dim: latent dimension
        conv_dim: base number of channels
        image_size: image resolution
        dist: generator distribution. Choices: deterministic, gaussian, implicit
        g_std: scaling the standard deviation of the gaussian generator. Default: 1
    '''
    def __init__(self, latent_dim=64, conv_dim=32, image_size=64, dist='deterministic', g_std=1):
        super().__init__()
        self.latent_dim = latent_dim
        self.dist = dist
        self.g_std = g_std

        out_channels = 6 if dist == 'gaussian' else 3
        add_noise = True if dist == 'implicit' else False
        self.decoder = Generator(latent_dim, conv_dim, image_size, out_channels, add_noise)

    def forward(self, z, mean=False, stats=False):
        out = self.decoder(z)
        if self.dist == 'gaussian':
            x_mu, x_logvar = out.chunk(2, dim=1)
            if stats:
                return x_mu, x_logvar
            else:
                x_sample = reparameterize(x_mu, (x_logvar / 2).exp(), self.g_std)
                if mean:
                    return x_mu
                else:
                    return x_sample
        else:
            return out


class BGM(nn.Module):
    r'''Bidirectional generative model

        Args:
            General settings:
                latent_dim: latent dimension
                conv_dim: base number of channels
                image_size: image resolution
                image_channel: number of image channel
            Encoder settings:
                enc_dist: encoder distribution
                enc_arch: encoder architecture
                enc_fc_size: number of nodes in each fc layer in encoder
                enc_noise_dim: dimension of input noise when an implicit encoder is used
            Generator settings:
                dec_dist: generator distribution. Choices: deterministic, implicit
                dec_arch: generator architecture. Choices: sagan, dcgan

    '''
    def __init__(self, latent_dim=64, conv_dim=32, image_size=64, image_channel=3,
                 enc_dist='gaussian', enc_arch='resnet', enc_fc_size=2048, enc_noise_dim=128,
                 dec_dist='deterministic',
                 type='big', old=False):
        super().__init__()
        self.latent_dim = latent_dim
        self.enc_dist = enc_dist
        self.dec_dist = dec_dist

        if type == 'big':
            self.encoder = ResEncoder(latent_dim, enc_arch, enc_dist, enc_fc_size, enc_noise_dim)
            if old:
                self.decoder = Generator(latent_dim, conv_dim, image_size)
            else:
                self.decoder = BigDecoder(latent_dim, conv_dim, image_size, dec_dist)
        elif type == 'dcgan':
            self.encoder = DCEncoder(latent_dim, conv_dim, image_size, image_channel, enc_dist)
            self.decoder = DCDecoder(latent_dim, conv_dim, image_size, image_channel, dec_dist)

    def forward(self, x=None, z=None, recon=False, infer_mean=True):
        # recon_mean is used for gaussian decoder which we do not use here.
        # Training Mode
        if x is not None and z is not None:
            if self.enc_dist == 'gaussian':
                z_mu, z_logvar = self.encoder(x)
                z_fake = reparameterize(z_mu, (z_logvar / 2).exp())
            else: # deterministic or implicit
                z_fake = self.encoder(x)
            x_fake = self.decoder(z)
            return z_fake, x_fake

        # Inference Mode
        elif x is not None and z is None:
            # Get latent
            if self.enc_dist == 'gaussian':
                z_mu, z_logvar = self.encoder(x)
                z_fake = reparameterize(z_mu, (z_logvar / 2).exp())
            else: # deterministic or implicit
                z_fake = self.encoder(x)

            # Reconstruction
            if recon:
                return self.decoder(z_fake)

            # Representation
            # Mean representation for Gaussian encoder
            elif infer_mean and self.enc_dist == 'gaussian':
                return z_mu
            # Random representation sampled from q_e(z|x)
            else:
                return z_fake

        # Generation Mode
        elif x is None and z is not None:
            return self.decoder(z)


class BigJointDiscriminator(nn.Module):
    r'''Big joint discriminator based on SAGAN

    Args:
        latent_dim: latent dimension
        conv_dim: base number of channels
        image_size: image resolution
        fc_size: number of nodes in each fc layers
    '''
    def __init__(self, latent_dim=64, conv_dim=32, image_size=64, fc_size=1024):
        super().__init__()
        self.discriminator = Discriminator(conv_dim, image_size, in_channels=3, out_feature=True)
        self.discriminator_z = Discriminator_MLP(latent_dim, fc_size)
        self.discriminator_j = Discriminator_MLP(conv_dim * 16 + fc_size, fc_size)

    def forward(self, x, z):
        sx, feature_x = self.discriminator(x)
        sz, feature_z = self.discriminator_z(z)
        sxz, _ = self.discriminator_j(torch.cat((feature_x, feature_z), dim=1))
        return (sx + sz + sxz) / 3


class DCAE(nn.Module):
    def __init__(self, latent_dim=64, conv_dim=64, image_size=28, image_channel=3, enc_dist='gaussian',
                 dec_dist='deterministic', tanh=False):
        super().__init__()
        self.latent_dim = latent_dim
        self.enc_dist = enc_dist
        self.dec_dist = dec_dist
        self.encoder = DCEncoder(latent_dim, conv_dim, image_size, image_channel, enc_dist, tanh=tanh)
        self.decoder = DCDecoder(latent_dim, conv_dim, image_size, image_channel, dec_dist)

    def forward(self, x=None, z=None, infer_mean=True, recon=False, gen_mean=True, recon_mean=False):
        # Training Mode (only used in age)
        if x is not None and z is not None:
            if self.enc_dist == 'gaussian':
                z_mu, z_logvar = self.encoder(x)
                z_fake = reparameterize(z_mu, (z_logvar / 2).exp())
            else: #deterministic
                z_fake = self.encoder(x)
            x_fake = self.decoder(z)
            return z_fake, x_fake

        # Inference Mode
        elif x is not None and z is None:
            if self.enc_dist == 'gaussian':
                z_mu, z_logvar = self.encoder(x)
            # Reconstruction
            if recon:
                if self.enc_dist == 'gaussian':
                    z_fake = reparameterize(z_mu, (z_logvar / 2).exp())
                else:
                    z_fake = self.encoder(x)
                if self.dec_dist == 'gaussian':
                    if recon_mean:
                        return self.decoder(z_fake, mean=True)
                    else:
                        x_recon, x_mu, x_logvar = self.decoder(z_fake, stats=True)
                        return x_recon, x_mu, x_logvar, z_mu, z_logvar
                else:
                    return self.decoder(z_fake)
            # Representation
            else:
                if self.enc_dist != 'gaussian':
                    return self.encoder(x)
                else:
                    if infer_mean:  # Mean representation
                        return z_mu
                    else:  # Sample representation
                        z_fake = reparameterize(z_mu, (z_logvar / 2).exp())
                        return z_fake

        # Generation Mode
        elif x is None and z is not None:
            x_fake = self.decoder(z, mean=gen_mean)
            return x_fake


class DCEncoder(nn.Module):
    '''DCGAN discriminator'''
    def __init__(self, latent_dim=64, conv_dim=64, image_size=28, image_channel=3, dist='gaussian', noise_dim=100, tanh=False):
        super().__init__()
        self.dist = dist
        in_channels = image_channel + noise_dim if dist == 'implicit' else image_channel
        self.noise_dim = noise_dim
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, conv_dim, 5, 2, 2),
            nn.BatchNorm2d(conv_dim * 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(conv_dim, conv_dim * 2, 5, 2, 2),
            nn.BatchNorm2d(conv_dim * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(conv_dim * 2, conv_dim * 4, 5, 2, 2),
            nn.BatchNorm2d(conv_dim * 4),
            nn.ReLU(inplace=True),
        )
        fc_size = latent_dim * 2 if dist == 'gaussian' else latent_dim
        self.fc = nn.Linear(conv_dim * 4 * 4 * 4, fc_size)
        self.add_tanh = tanh
        if tanh:
            self.tanh = nn.Tanh()

    def forward(self, x):
        if self.dist == 'implicit':
            eps = torch.randn(x.size(0), self.noise_dim, device=x.device)
            eps = eps.view(x.size(0), self.noise_dim, 1, 1).expand(x.size(0), self.noise_dim, x.size(2), x.size(2))
            x = torch.cat([x, eps], dim=1)
        x = self.conv(x).view(x.size(0), -1)
        if self.dist == 'gaussian':
            return self.fc(x).chunk(2, dim=1)
        else:
            if self.add_tanh:
                return self.tanh(self.fc(x))
            else:
                return self.fc(x)


class DCDecoder(nn.Module):
    '''DCGAN Generator'''
    def __init__(self, latent_dim=64, conv_dim=64, image_size=28, image_channel=3, dist='deterministic'):
        super().__init__()
        self.dist = dist
        self.conv_dim = conv_dim
        # Input 100
        if dist == 'implicit':
            self.fc = nn.Linear(latent_dim, conv_dim * 4 * 4 * 4)
            self.bn0 = nn.BatchNorm2d(conv_dim * 4)
            self.conv2 = nn.ConvTranspose2d(conv_dim * 4, conv_dim * 2, 5, 2, 2)
            self.bn2 = nn.BatchNorm2d(conv_dim * 2)
            self.conv3 = nn.ConvTranspose2d(conv_dim * 2, conv_dim, 5, 2, 2, 1)
            self.bn3 = nn.BatchNorm2d(conv_dim)
            self.toRGB = nn.ConvTranspose2d(conv_dim, image_channel, 5, 2, 2, 1)
            self.relu = nn.ReLU(True)
            self.tanh = nn.Tanh()
            self.noise1 = NoiseInjection(conv_dim * 4, 4)
            self.noise2 = NoiseInjection(conv_dim * 2, 7)
            self.noise3 = NoiseInjection(conv_dim, 14)

        else:
            out_channels = image_channel if dist == 'deterministic' else image_channel * 2
            self.fc = nn.Sequential(
                nn.Linear(latent_dim, conv_dim * 4 * 4 * 4),
                nn.BatchNorm1d(conv_dim * 4 * 4 * 4),
                nn.ReLU(True)
            )
            self.net = nn.Sequential(
               nn.ConvTranspose2d(conv_dim * 4, conv_dim * 2, 5, 2, 2),
                nn.BatchNorm2d(conv_dim * 2),
                nn.ReLU(True),
                nn.ConvTranspose2d(conv_dim * 2, conv_dim, 5, 2, 2, 1),
                nn.BatchNorm2d(conv_dim),
                nn.ReLU(True),
                nn.ConvTranspose2d(conv_dim, out_channels, 5, 2, 2, 1),
                nn.Tanh(),
            )

    def forward(self, z, mean=False, stats=False):
        out = self.fc(z).view(-1, self.conv_dim * 4, 4, 4)
        if self.dist == 'gaussian':
            x_mu, x_logvar = self.net(out).chunk(2, dim=1)
            x_sample = reparameterize(x_mu, (x_logvar / 2).exp())
            if stats:
                return x_sample, x_mu, x_logvar
            else:
                if mean:
                    return x_mu
                else:
                    return x_sample
        elif self.dist == 'implicit':
            out = self.relu(self.bn0(self.noise1(out)))
            out = self.relu(self.bn2(self.noise2(self.conv2(out))))
            out = self.relu(self.bn3(self.noise3(self.conv3(out))))
            return self.tanh(self.toRGB(out))
        else:
            return self.net(out)


class DCJointDiscriminator(nn.Module):
    def __init__(self, latent_dim=64, conv_dim=64, image_size=64, image_channel=3, fc_dim=1024):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(image_channel, conv_dim, 5, 2, 2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(conv_dim, conv_dim * 2, 5, 2, 2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(conv_dim * 2, conv_dim * 4, 5, 2, 2),
            nn.LeakyReLU(inplace=True),
        )
        self.fc_j = nn.Sequential(
            nn.Linear(conv_dim * 4 * 4 * 4 + latent_dim, fc_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(fc_dim, 1)
        )

    def forward(self, x, z):
        x = self.conv(x).view(x.size(0), -1)
        xz = torch.cat([x, z], dim=1)
        sxz = self.fc_j(xz)
        return sxz


class ToyAE(nn.Module):
    def __init__(self, data_dim=2, latent_dim=10, enc_hidden_dim=500, dec_hidden_dim=500,
                 enc_dist='gaussian', dec_dist='gaussian'):
        super().__init__()
        self.latent_dim = latent_dim
        self.enc_dist = enc_dist
        self.dec_dist = dec_dist
        self.encoder = EncoderMLP(data_dim, latent_dim, enc_hidden_dim, enc_dist)
        self.decoder = DecoderMLP(data_dim, latent_dim, dec_hidden_dim, dec_dist)

    def forward(self, x=None, z=None, infer_mean=True, recon=False, gen_mean=False):
        # Training Mode (only used in age)
        if x is not None and z is not None:
            if self.enc_dist == 'gaussian':
                z_mu, z_logvar = self.encoder(x)
                z_fake = reparameterize(z_mu, (z_logvar / 2).exp())
            else: # deterministic or implicit
                z_fake = self.encoder(x)
            x_fake = self.decoder(z)
            return z_fake, x_fake

        # Inference Mode
        elif x is not None and z is None:
            if self.enc_dist == 'gaussian':
                z_mu, z_logvar = self.encoder(x)
                z_fake = reparameterize(z_mu, (z_logvar / 2).exp())
            else: # deterministic or implicit
                z_fake = self.encoder(x)

            # Reconstruction
            if recon:
                if self.dec_dist == 'gaussian':
                    x_recon, x_mu, x_logvar = self.decoder(z_fake, stats=True)
                    return x_recon, x_mu, x_logvar, z_mu, z_logvar
                else:
                    return self.decoder(z_fake)
            # Representation
            elif infer_mean and self.enc_dist == 'gaussian': # Mean representation
                return z_mu
            else: # Sample representation
                return z_fake

        # Generation Mode
        elif x is None and z is not None:
            x_fake = self.decoder(z, mean=gen_mean)
            return x_fake


class DecoderMLP(nn.Module):
    def __init__(self, data_dim=2, latent_dim=10, hidden_dim=500, dist='gaussian'):
        super().__init__()
        self.dist = dist
        net = [
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        ]
        if dist == 'gaussian':
            net.append(nn.Linear(hidden_dim, data_dim * 2))
        else:
            assert self.dist == 'deterministic'
            net.append(nn.Linear(hidden_dim, data_dim))
        self.decoder = nn.Sequential(*net)

    def forward(self, z, mean=False, stats=False):
        if self.dist == 'gaussian':
            x_mu, x_logvar = self.decoder(z).chunk(2, dim=1)
            x_sample = reparameterize(x_mu, (x_logvar / 2).exp())
            if stats:
                return x_sample, x_mu, x_logvar
            else:
                if mean:
                    return x_mu
                else:
                    return x_sample
        else:
            return self.decoder(z)


class EncoderMLP(nn.Module):
    def __init__(self, data_dim=2, latent_dim=10, hidden_dim=500, dist='gaussian'):
        super().__init__()
        self.dist = dist
        net = [
            nn.Linear(data_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        ]
        if dist == 'gaussian':
            net.append(nn.Linear(hidden_dim, latent_dim * 2))
        else:
            assert self.dist == 'deterministic'
            net.append(nn.Linear(hidden_dim, latent_dim))
        self.encoder = nn.Sequential(*net)

    def forward(self, x):
        if self.dist == 'gaussian':
            return self.encoder(x).chunk(2, dim=1)
        else:
            return self.encoder(x)


class DiscriminatorMLP(nn.Module):
    def __init__(self, data_dim=2, latent_dim=10, hidden_dim_x=400, hidden_dim_z=500, hidden_dim=500):
        super().__init__()
        self.dis_z = DisMLPBlock(latent_dim, hidden_dim_z)
        self.dis_x = DisMLPBlock(data_dim, hidden_dim_x)
        self.dis_j = DisMLPBlock(hidden_dim_z + hidden_dim_x, hidden_dim)

    def forward(self, x, z):
        sz, fz = self.dis_z(z)
        sx, fx = self.dis_x(x)
        sj, _ = self.dis_j(torch.cat([fz, fx], dim=1))
        return (sz + sx + sj) / 3


class DisMLPBlock(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=500):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(inplace=True),
        )
        self.layer = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        feature = self.block(x)
        return self.layer(feature), feature


def reparameterize(mu, sigma, std=1):
    assert mu.shape == sigma.shape
    eps = mu.new(mu.shape).normal_(0, std)
    return mu + sigma * eps

def kl_div(mu, logvar):
    return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum()

def gaussian_nll(x_mu, x_logvar, x):
    '''NLL'''
    sigma_inv = (- x_logvar / 2).exp()
    return 0.5 * (x_logvar + ((x - x_mu) * sigma_inv).pow(2) + np.log(2*np.pi)).sum()

def kaiming_init(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif type(m) == nn.BatchNorm1d or type(m) == nn.BatchNorm2d:
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)

