import torch
import numpy as np
from torch import nn
from torch.nn import functional as F

LRELU_SLOPE = 0.1


class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        x = x.transpose(1, -1).contiguous()
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1).contiguous()


class ConvReluNorm(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, kernel_size,
                 n_layers, p_dropout):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.p_dropout = p_dropout
        assert n_layers > 1, "Number of layers should be larger than 0."

        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.conv_layers.append(
            nn.Conv1d(in_channels, hidden_channels, kernel_size,
                      padding=kernel_size // 2))
        self.norm_layers.append(LayerNorm(hidden_channels))
        self.relu_drop = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(p_dropout))
        for _ in range(n_layers - 1):
            self.conv_layers.append(
                nn.Conv1d(hidden_channels, hidden_channels, kernel_size,
                          padding=kernel_size // 2))
            self.norm_layers.append(LayerNorm(hidden_channels))
        self.proj = nn.Conv1d(hidden_channels, out_channels, 1)
        self.proj.weight.data.zero_()
        self.proj.bias.data.zero_()

    def forward(self, x, x_mask):
        x_org = x
        for i in range(self.n_layers):
            x = self.conv_layers[i](x * x_mask)
            x = self.norm_layers[i](x)
            x = self.relu_drop(x)
        x = x_org + self.proj(x)
        return x * x_mask


class DDSConv(nn.Module):
    """
    Dialted and Depth-Separable Convolution
    """

    def __init__(self, channels, kernel_size, n_layers, p_dropout=0.):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.p_dropout = p_dropout

        self.drop = nn.Dropout(p_dropout)
        self.convs_sep = nn.ModuleList()
        self.convs_1x1 = nn.ModuleList()
        self.norms_1 = nn.ModuleList()
        self.norms_2 = nn.ModuleList()
        for i in range(n_layers):
            dilation = kernel_size ** i
            padding = (kernel_size * dilation - dilation) // 2
            self.convs_sep.append(nn.Conv1d(channels, channels, kernel_size,
                                            groups=channels, dilation=dilation,
                                            padding=padding
                                            ))
            self.convs_1x1.append(nn.Conv1d(channels, channels, 1))
            self.norms_1.append(LayerNorm(channels))
            self.norms_2.append(LayerNorm(channels))

    def forward(self, x, x_mask, g=None):
        if g is not None:
            x = x + g
        for i in range(self.n_layers):
            y = self.convs_sep[i](x * x_mask)
            y = self.norms_1[i](y)
            y = F.gelu(y)
            y = self.convs_1x1[i](y)
            y = self.norms_2[i](y)
            y = F.gelu(y)
            y = self.drop(y)
            x = x + y
        return x * x_mask


class Log(nn.Module):
    def forward(self, x, x_mask, reverse=False, **kwargs):
        if not reverse:
            y = torch.log(torch.clamp_min(x, 1e-5)) * x_mask
            logdet = torch.sum(-y, [1, 2])
            return y, logdet
        else:
            x = torch.exp(x) * x_mask
            return x


class Flip(nn.Module):
    def forward(self, x, *args, reverse=False, **kwargs):
        x = torch.flip(x, [1])
        if not reverse:
            logdet = torch.zeros(x.size(0)).to(dtype=x.dtype, device=x.device)
            return x, logdet
        else:
            return x


class ElementwiseAffine(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.m = nn.Parameter(torch.zeros(channels, 1))
        self.logs = nn.Parameter(torch.zeros(channels, 1))

    def forward(self, x, x_mask, reverse=False, **kwargs):
        if not reverse:
            y = self.m + torch.exp(self.logs) * x
            y = y * x_mask
            logdet = torch.sum(self.logs * x_mask, [1, 2])
            return y, logdet
        else:
            x = (x - self.m) * torch.exp(-self.logs) * x_mask
            return x


class LinearNorm(nn.Module):
  def __init__(self,
                in_channels,
                out_channels,
                bias=True,
                spectral_norm=False,
                ):
      super(LinearNorm, self).__init__()
      self.fc = nn.Linear(in_channels, out_channels, bias)

      if spectral_norm:
          self.fc = nn.utils.spectral_norm(self.fc)

  def forward(self, input):
      out = self.fc(input)
      return out


class Mish(nn.Module):
  def __init__(self):
      super(Mish, self).__init__()
  def forward(self, x):
      return x * torch.tanh(F.softplus(x))


class Conv1dGLU(nn.Module):
  '''
  Conv1d + GLU(Gated Linear Unit) with residual connection.
  For GLU refer to https://arxiv.org/abs/1612.08083 paper.
  '''
  def __init__(self, in_channels, out_channels, kernel_size, dropout):
      super(Conv1dGLU, self).__init__()
      self.out_channels = out_channels
      self.conv1 = ConvNorm(in_channels, 2*out_channels, kernel_size=kernel_size)
      self.dropout = nn.Dropout(dropout)

  def forward(self, x):
      residual = x
      x = self.conv1(x)
      x1, x2 = torch.split(x, split_size_or_sections=self.out_channels, dim=1)
      x = x1 * torch.sigmoid(x2)
      x = residual + self.dropout(x)
      return x


class ConvNorm(nn.Module):
  def __init__(self,
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=None,
                dilation=1,
                bias=True,
                spectral_norm=False,
                ):
      super(ConvNorm, self).__init__()

      if padding is None:
          assert(kernel_size % 2 == 1)
          padding = int(dilation * (kernel_size - 1) / 2)

      self.conv = torch.nn.Conv1d(in_channels,
                                  out_channels,
                                  kernel_size=kernel_size,
                                  stride=stride,
                                  padding=padding,
                                  dilation=dilation,
                                  bias=bias)

      if spectral_norm:
          self.conv = nn.utils.spectral_norm(self.conv)

  def forward(self, input):
      out = self.conv(input)
      return out


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''
    def __init__(self, n_head, d_model, d_k, d_v,
                 dropout=0., spectral_norm=False):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)

        self.attention = ScaledDotProductAttention(
            temperature=np.power(d_model, 0.5), dropout=dropout)

        self.fc = nn.Linear(n_head * d_v, d_model)
        self.dropout = nn.Dropout(dropout)

        if spectral_norm:
            self.w_qs = nn.utils.spectral_norm(self.w_qs)
            self.w_ks = nn.utils.spectral_norm(self.w_ks)
            self.w_vs = nn.utils.spectral_norm(self.w_vs)
            self.fc = nn.utils.spectral_norm(self.fc)

    def forward(self, x, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_x, _ = x.size()

        residual = x

        q = self.w_qs(x).view(sz_b, len_x, n_head, d_k)
        k = self.w_ks(x).view(sz_b, len_x, n_head, d_k)
        v = self.w_vs(x).view(sz_b, len_x, n_head, d_v)
        q = q.permute(2, 0, 1, 3).contiguous().view(
            -1, len_x, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(
            -1, len_x, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(
            -1, len_x, d_v)  # (n*b) x lv x dv

        if mask is not None:
            slf_mask = mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..
        else:
            slf_mask = None
        output, attn = self.attention(q, k, v, mask=slf_mask)

        output = output.view(n_head, sz_b, len_x, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(
                        sz_b, len_x, -1)  # b x lq x (n*dv)

        output = self.fc(output)

        output = self.dropout(output) + residual
        return output, attn


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, dropout):
        super().__init__()
        self.temperature = temperature
        self.softmax = nn.Softmax(dim=2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        p_attn = self.dropout(attn)

        output = torch.bmm(p_attn, v)
        return output, attn


class AffineLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(AffineLinear, self).__init__()
        affine = nn.Linear(in_dim, out_dim)
        self.affine = affine

    def forward(self, input):
        return self.affine(input)


class StyleAdaptiveLayerNorm(nn.Module):
    def __init__(self, in_channel, style_dim):
        super(StyleAdaptiveLayerNorm, self).__init__()
        self.in_channel = in_channel
        self.norm = nn.LayerNorm(in_channel, elementwise_affine=False)

        self.style = AffineLinear(style_dim, in_channel * 2)
        self.style.affine.bias.data[:in_channel] = 1
        self.style.affine.bias.data[in_channel:] = 0

    def forward(self, input, style_code):
        """
        :param input: B T D1
        :param style_code: B 1 D2
        :return:  B T D1
        """
        style = self.style(style_code)   # B 1 2*D1
        gamma, beta = style.chunk(2, dim=-1)  # B 1 D1
        out = self.norm(input)  # B T D1
        out = gamma * out + beta
        return out


class MultiHeadAttention_GivenK(nn.Module):
    '''
    input:
        query --- [N, T_q, query_dim]
        key --- [N, T_k, key_dim]
    output:
        out --- [N, T_q, num_units]
    '''

    def __init__(self, query_dim, key_dim, num_units, num_heads):

        super().__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.key_dim = key_dim

        self.W_query = nn.Linear(
            in_features=query_dim, out_features=num_units, bias=False)
        self.W_key = nn.Linear(
            in_features=key_dim, out_features=num_units, bias=False)
        self.W_value = nn.Linear(
            in_features=key_dim, out_features=num_units, bias=False)

    def forward(self, query, key):
        querys = self.W_query(query)  # [N, T_q, num_units]
        keys = self.W_key(key)  # [N, T_k, num_units]
        values = self.W_value(key)

        split_size = self.num_units // self.num_heads
        # [h, N, T_q, num_units/h]
        querys = torch.stack(torch.split(querys, split_size, dim=2), dim=0)
        # [h, N, T_k, num_units/h]
        keys = torch.stack(torch.split(keys, split_size, dim=2), dim=0)
        # [h, N, T_k, num_units/h]
        values = torch.stack(torch.split(values, split_size, dim=2), dim=0)

        # score = softmax(QK^T / (d_k ** 0.5))
        scores = torch.matmul(querys, keys.transpose(2, 3))  # [h, N, T_q, T_k]
        scores = scores / (self.key_dim ** 0.5)
        scores = F.softmax(scores, dim=3)

        # out = score * V
        out = torch.matmul(scores, values)  # [h, N, T_q, num_units/h]
        # [N, T_q, num_units]
        out = torch.cat(torch.split(out, 1, dim=0), dim=3).squeeze(0)

        return out