# RG_SA自注意力模块
class RG_SA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., c_ratio=0.5):
        super(RG_SA, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.cr = int(dim * c_ratio)
        self.scale = qk_scale or (head_dim * c_ratio) ** -0.5
        self.reduction1 = nn.Conv2d(dim, dim, kernel_size=4, stride=4, groups=dim)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
        self.conv = nn.Conv2d(dim, self.cr, kernel_size=1, stride=1)
        self.norm_act = nn.Sequential(nn.LayerNorm(self.cr), nn.GELU())
        self.q = nn.Linear(dim, self.cr, bias=qkv_bias)
        self.k = nn.Linear(self.cr, self.cr, bias=qkv_bias)
        self.v = nn.Linear(self.cr, dim, bias=qkv_bias)
        self.cpe = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, H, W):
        B, N, C = x.shape
        _scale = 1
        _x = x.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
        if self.training:
            _time = max(int(math.log(H // 4, 4)), int(math.log(W // 4, 4)))
        else:
            _time = max(int(math.log(H // 16, 4)), int(math.log(W // 16, 4)))
            if _time < 2: _time = 2
        _scale = 4 ** _time
        for _ in range(_time):
            _x = self.reduction1(_x)
        _x = self.conv(self.dwconv(_x)).reshape(B, self.cr, -1).permute(0, 2, 1).contiguous()
        _x = self.norm_act(_x)
        q = self.q(x).reshape(B, N, self.num_heads, int(self.cr / self.num_heads)).permute(0, 2, 1, 3)
        k = self.k(_x).reshape(B, -1, self.num_heads, int(self.cr / self.num_heads)).permute(0, 2, 1, 3)
        v = self.v(_x).reshape(B, -1, self.num_heads, int(C / self.num_heads)).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        v = v + self.cpe(v.transpose(1, 2).reshape(B, -1, C).transpose(1, 2).contiguous().view(B, C, H // _scale, W // _scale)).view(B, C, -1).view(B, self.num_heads, int(C / self.num_heads), -1).transpose(-1, -2)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
