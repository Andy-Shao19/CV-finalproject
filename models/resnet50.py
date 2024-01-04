import torch.nn as nn
import torch
import torchvision.models as models


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def build_resnet50():
    resnet = models.resnet50(pretrained=True)
    return resnet


class ResNet50(nn.Module):
    def __init__(self, checkpoint):
        super(ResNet50, self).__init__()
        resnet = build_resnet50()
        if checkpoint:
            resnet.load_state_dict(torch.load(checkpoint))
        
        # 提取 ResNet50 的各个层
        self.enc_1 = nn.Sequential(*list(resnet.children())[:3])  # 初始层到第一个残差块
        self.enc_2 = nn.Sequential(*list(resnet.children())[3:5])  # 第一个残差块到第二个
        self.enc_3 = nn.Sequential(*list(resnet.children())[5:6])  # 第二个残差块到第三个
        self.enc_4 = nn.Sequential(*list(resnet.children())[6:7])  # 第三个残差块到第四个

        # fix the encoder
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

        self.mse_loss = nn.MSELoss()

    # extract relu1_1 - relu(n_layer)_1 from input image
    def encode_with_intermediate(self, x, n_layer=4):
        results = [x]
        for i in range(n_layer):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    # extract relu(n_layer)_1 from input image
    def encode(self, x, n_layer=4):
        for i in range(n_layer):
            x = getattr(self, 'enc_{:d}'.format(i + 1))(x)
        return x

    def calc_content_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        return self.mse_loss(input, target)

    def calc_style_loss(self, input, target):
        # assert (input.size() == target.size())
        assert (target.requires_grad is False)
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + self.mse_loss(input_std, target_std)

    def forward(self, content_images, style_images, stylized_images, n_layer=4, content_weight=0):

        style_feats = self.encode_with_intermediate(style_images, n_layer)
        stylized_feats = self.encode_with_intermediate(stylized_images, n_layer)

        # content loss

        if content_weight > 0:
            content_feat = self.encode(content_images)
            loss_c = self.calc_content_loss(stylized_feats[3], content_feat)    # relu4_1
        else:
            loss_c = 0

        # style loss
        loss_s = self.calc_style_loss(stylized_feats[0], style_feats[0])
        for i in range(1, n_layer):
            loss_s += self.calc_style_loss(stylized_feats[i], style_feats[i])

        return loss_c, loss_s



