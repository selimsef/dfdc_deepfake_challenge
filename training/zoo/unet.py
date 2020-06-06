from functools import partial

import torch
from timm.models.efficientnet import tf_efficientnet_b3_ns, tf_efficientnet_b5_ns
from torch import nn
from torch.nn import Dropout2d, Conv2d
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.pooling import AdaptiveAvgPool2d
from torch.nn.modules.upsampling import UpsamplingBilinear2d

encoder_params = {
    "tf_efficientnet_b3_ns": {
        "features": 1536,
        "filters": [40, 32, 48, 136, 1536],
        "decoder_filters": [64, 128, 256, 256],
        "init_op": partial(tf_efficientnet_b3_ns, pretrained=True, drop_path_rate=0.2)
    },
    "tf_efficientnet_b5_ns": {
        "features": 2048,
        "filters": [48, 40, 64, 176, 2048],
        "decoder_filters": [64, 128, 256, 256],
        "init_op": partial(tf_efficientnet_b5_ns, pretrained=True, drop_path_rate=0.2)
    },
}


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layer(x)


class ConcatBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, dec, enc):
        x = torch.cat([dec, enc], dim=1)
        return self.seq(x)


class Decoder(nn.Module):
    def __init__(self, decoder_filters, filters, upsample_filters=None,
                 decoder_block=DecoderBlock, bottleneck=ConcatBottleneck, dropout=0):
        super().__init__()
        self.decoder_filters = decoder_filters
        self.filters = filters
        self.decoder_block = decoder_block
        self.decoder_stages = nn.ModuleList([self._get_decoder(idx) for idx in range(0, len(decoder_filters))])
        self.bottlenecks = nn.ModuleList([bottleneck(self.filters[-i - 2] + f, f)
                                          for i, f in enumerate(reversed(decoder_filters))])
        self.dropout = Dropout2d(dropout) if dropout > 0 else None
        self.last_block = None
        if upsample_filters:
            self.last_block = decoder_block(decoder_filters[0], out_channels=upsample_filters)
        else:
            self.last_block = UpsamplingBilinear2d(scale_factor=2)

    def forward(self, encoder_results: list):
        x = encoder_results[0]
        bottlenecks = self.bottlenecks
        for idx, bottleneck in enumerate(bottlenecks):
            rev_idx = - (idx + 1)
            x = self.decoder_stages[rev_idx](x)
            x = bottleneck(x, encoder_results[-rev_idx])
        if self.last_block:
            x = self.last_block(x)
        if self.dropout:
            x = self.dropout(x)
        return x

    def _get_decoder(self, layer):
        idx = layer + 1
        if idx == len(self.decoder_filters):
            in_channels = self.filters[idx]
        else:
            in_channels = self.decoder_filters[idx]
        return self.decoder_block(in_channels, self.decoder_filters[max(layer, 0)])


def _initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
            m.weight.data = nn.init.kaiming_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


class EfficientUnetClassifier(nn.Module):
    def __init__(self, encoder, dropout_rate=0.5) -> None:
        super().__init__()
        self.decoder = Decoder(decoder_filters=encoder_params[encoder]["decoder_filters"],
                               filters=encoder_params[encoder]["filters"])
        self.avg_pool = AdaptiveAvgPool2d((1, 1))
        self.dropout = Dropout(dropout_rate)
        self.fc = Linear(encoder_params[encoder]["features"], 1)
        self.final = Conv2d(encoder_params[encoder]["decoder_filters"][0], out_channels=1, kernel_size=1, bias=False)
        _initialize_weights(self)
        self.encoder = encoder_params[encoder]["init_op"]()

    def get_encoder_features(self, x):
        encoder_results = []
        x = self.encoder.conv_stem(x)
        x = self.encoder.bn1(x)
        x = self.encoder.act1(x)
        encoder_results.append(x)
        x = self.encoder.blocks[:2](x)
        encoder_results.append(x)
        x = self.encoder.blocks[2:3](x)
        encoder_results.append(x)
        x = self.encoder.blocks[3:5](x)
        encoder_results.append(x)
        x = self.encoder.blocks[5:](x)
        x = self.encoder.conv_head(x)
        x = self.encoder.bn2(x)
        x = self.encoder.act2(x)
        encoder_results.append(x)
        encoder_results = list(reversed(encoder_results))
        return encoder_results

    def forward(self, x):
        encoder_results = self.get_encoder_features(x)
        seg = self.final(self.decoder(encoder_results))
        x = encoder_results[0]
        x = self.avg_pool(x).flatten(1)
        x = self.dropout(x)
        x = self.fc(x)
        return x, seg


if __name__ == '__main__':
    model = EfficientUnetClassifier("tf_efficientnet_b5_ns")
    model.eval()
    with torch.no_grad():
        input = torch.rand(4, 3, 224, 224)
        print(model(input))
