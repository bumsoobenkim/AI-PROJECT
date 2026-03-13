"""CycleGAN: Unpaired image-to-image translation (Zhu et al., ICCV 2017).
   Pix2Pix: Paired conditional image synthesis (Isola et al., CVPR 2017)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 3), nn.InstanceNorm2d(dim), nn.ReLU(True),
            nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 3), nn.InstanceNorm2d(dim)
        )
    def forward(self, x): return x + self.net(x)


class CycleGANGenerator(nn.Module):
    """ResNet-based generator for CycleGAN."""
    def __init__(self, in_channels=3, out_channels=3, ngf=64, n_res=9):
        super().__init__()
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, ngf, 7), nn.InstanceNorm2d(ngf), nn.ReLU(True),
            nn.Conv2d(ngf, ngf*2, 3, 2, 1), nn.InstanceNorm2d(ngf*2), nn.ReLU(True),
            nn.Conv2d(ngf*2, ngf*4, 3, 2, 1), nn.InstanceNorm2d(ngf*4), nn.ReLU(True),
        ]
        layers += [ResBlock(ngf*4) for _ in range(n_res)]
        layers += [
            nn.ConvTranspose2d(ngf*4, ngf*2, 3, 2, 1, 1), nn.InstanceNorm2d(ngf*2), nn.ReLU(True),
            nn.ConvTranspose2d(ngf*2, ngf, 3, 2, 1, 1), nn.InstanceNorm2d(ngf), nn.ReLU(True),
            nn.ReflectionPad2d(3), nn.Conv2d(ngf, out_channels, 7), nn.Tanh()
        ]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)


class PatchGANDiscriminator(nn.Module):
    """PatchGAN discriminator: classify overlapping patches as real/fake."""
    def __init__(self, in_channels=3, ndf=64, n_layers=3):
        super().__init__()
        layers = [nn.Conv2d(in_channels, ndf, 4, 2, 1), nn.LeakyReLU(0.2, True)]
        nf = ndf
        for i in range(1, n_layers):
            nf_prev = nf; nf = min(nf * 2, 512)
            layers += [nn.Conv2d(nf_prev, nf, 4, 2, 1), nn.InstanceNorm2d(nf), nn.LeakyReLU(0.2, True)]
        layers += [nn.Conv2d(nf, min(nf*2,512), 4, 1, 1), nn.InstanceNorm2d(min(nf*2,512)), nn.LeakyReLU(0.2, True)]
        layers += [nn.Conv2d(min(nf*2,512), 1, 4, 1, 1)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)


class CycleGAN(nn.Module):
    """Full CycleGAN with two generators and two discriminators."""
    def __init__(self, ngf=64, ndf=64, lambda_cycle=10.0, lambda_idt=5.0):
        super().__init__()
        self.G_AB = CycleGANGenerator(ngf=ngf)
        self.G_BA = CycleGANGenerator(ngf=ngf)
        self.D_A = PatchGANDiscriminator(ndf=ndf)
        self.D_B = PatchGANDiscriminator(ndf=ndf)
        self.lambda_cycle = lambda_cycle
        self.lambda_idt = lambda_idt

    def cycle_consistency_loss(self, real_A, real_B):
        fake_B = self.G_AB(real_A)
        rec_A  = self.G_BA(fake_B)
        fake_A = self.G_BA(real_B)
        rec_B  = self.G_AB(fake_A)
        loss = F.l1_loss(rec_A, real_A) + F.l1_loss(rec_B, real_B)
        return loss * self.lambda_cycle

    def identity_loss(self, real_A, real_B):
        idt_A = self.G_BA(real_A)
        idt_B = self.G_AB(real_B)
        return (F.l1_loss(idt_A, real_A) + F.l1_loss(idt_B, real_B)) * self.lambda_idt

    def generator_loss(self, real_A, real_B):
        fake_B = self.G_AB(real_A); fake_A = self.G_BA(real_B)
        adv = F.mse_loss(self.D_B(fake_B), torch.ones_like(self.D_B(fake_B))) + \
              F.mse_loss(self.D_A(fake_A), torch.ones_like(self.D_A(fake_A)))
        return adv + self.cycle_consistency_loss(real_A, real_B) + self.identity_loss(real_A, real_B)

    def discriminator_loss(self, real_A, real_B):
        fake_B = self.G_AB(real_A).detach(); fake_A = self.G_BA(real_B).detach()
        real_label = torch.ones; fake_label = torch.zeros
        loss_A = 0.5*(F.mse_loss(self.D_A(real_A), torch.ones_like(self.D_A(real_A))) +
                      F.mse_loss(self.D_A(fake_A), torch.zeros_like(self.D_A(fake_A))))
        loss_B = 0.5*(F.mse_loss(self.D_B(real_B), torch.ones_like(self.D_B(real_B))) +
                      F.mse_loss(self.D_B(fake_B), torch.zeros_like(self.D_B(fake_B))))
        return loss_A + loss_B


class Pix2Pix(nn.Module):
    """Pix2Pix: paired image-to-image with U-Net generator + PatchGAN."""
    def __init__(self, ngf=64, ndf=64, lambda_l1=100.0):
        super().__init__()
        self.G = UNetGenerator(ngf=ngf)
        self.D = PatchGANDiscriminator(in_channels=6, ndf=ndf)
        self.lambda_l1 = lambda_l1

    def generator_loss(self, real_A, real_B):
        fake_B = self.G(real_A)
        d_fake = self.D(torch.cat([real_A, fake_B], dim=1))
        adv = F.mse_loss(d_fake, torch.ones_like(d_fake))
        l1 = F.l1_loss(fake_B, real_B) * self.lambda_l1
        return adv + l1, fake_B

    def discriminator_loss(self, real_A, real_B, fake_B):
        d_real = self.D(torch.cat([real_A, real_B], dim=1))
        d_fake = self.D(torch.cat([real_A, fake_B.detach()], dim=1))
        return 0.5*(F.mse_loss(d_real, torch.ones_like(d_real)) +
                    F.mse_loss(d_fake, torch.zeros_like(d_fake)))


class UNetGenerator(nn.Module):
    """U-Net generator for Pix2Pix."""
    def __init__(self, in_ch=3, out_ch=3, ngf=64):
        super().__init__()
        def down(ic, oc, norm=True):
            layers = [nn.Conv2d(ic, oc, 4, 2, 1, bias=not norm)]
            if norm: layers.append(nn.BatchNorm2d(oc))
            layers.append(nn.LeakyReLU(0.2, True))
            return nn.Sequential(*layers)
        def up(ic, oc, drop=False):
            layers = [nn.ConvTranspose2d(ic, oc, 4, 2, 1, bias=False), nn.BatchNorm2d(oc)]
            if drop: layers.append(nn.Dropout(0.5))
            layers.append(nn.ReLU(True))
            return nn.Sequential(*layers)
        self.d1=down(in_ch,ngf,False); self.d2=down(ngf,ngf*2)
        self.d3=down(ngf*2,ngf*4); self.d4=down(ngf*4,ngf*8)
        self.d5=down(ngf*8,ngf*8); self.d6=down(ngf*8,ngf*8)
        self.d7=down(ngf*8,ngf*8)
        self.bottleneck=nn.Sequential(nn.Conv2d(ngf*8,ngf*8,4,2,1),nn.ReLU(True))
        self.u1=up(ngf*8,ngf*8,True); self.u2=up(ngf*16,ngf*8,True)
        self.u3=up(ngf*16,ngf*8,True); self.u4=up(ngf*16,ngf*8)
        self.u5=up(ngf*16,ngf*4); self.u6=up(ngf*8,ngf*2); self.u7=up(ngf*4,ngf)
        self.final=nn.Sequential(nn.ConvTranspose2d(ngf*2,out_ch,4,2,1),nn.Tanh())

    def forward(self, x):
        d1=self.d1(x); d2=self.d2(d1); d3=self.d3(d2); d4=self.d4(d3)
        d5=self.d5(d4); d6=self.d6(d5); d7=self.d7(d6)
        b=self.bottleneck(d7)
        u1=self.u1(b); u2=self.u2(torch.cat([u1,d7],1)); u3=self.u3(torch.cat([u2,d6],1))
        u4=self.u4(torch.cat([u3,d5],1)); u5=self.u5(torch.cat([u4,d4],1))
        u6=self.u6(torch.cat([u5,d3],1)); u7=self.u7(torch.cat([u6,d2],1))
        return self.final(torch.cat([u7,d1],1))
