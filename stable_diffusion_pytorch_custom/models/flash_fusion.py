import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimpo

class FlashFusion(nn.Module):
    """Some Information about FlashFusion"""
    def __init__(self):
        super(FlashFusion, self).__init__()
        self.time_embedding = TimeEmbedding()

        self.entry_conv = PaddedConv2D(4,320)

        self.downblocks = [
            [
                nn.ModuleList([ResBlock(320,320),SpatialTransformer(320,8,40)]) for i in range(2)
            ],
            [PaddedConv2D(320,320,stride=2)],
            [
                nn.ModuleList([ResBlock(320,640),SpatialTransformer(640,8,80)]) for i in range(2)
            ],
            [PaddedConv2D(640,640,stride=2)],
            [
                nn.ModuleList([ResBlock(640,1280),SpatialTransformer(1280,8,160)]) for i in range(2)
            ],
            [PaddedConv2D(1280,1280,stride=2)],
            [
                nn.ModuleList([ResBlock(1280,1280)]) for i in range(2)
            ]
        ]

        self.middleblocks = nn.Sequential(            
                ResBlock(1280,1280),SpatialTransformer(1280,8,160),ResBlock(1280,1280)            
        )

        self.upblocks = [
            [
                nn.ModuleList([Concatenate(),ResBlock(1280*2,1280)]) for i in range(3)
            ],
            [
                Upsample(1280,1280)
            ],
            [
                nn.ModuleList([Concatenate(),ResBlock(1280*2,1280),SpatialTransformer(1280,8,160)]) for i in range(3)
            ],
            [
                Upsample(1280,1280)
            ],
            [
                nn.ModuleList([Concatenate(),ResBlock(1280*2,640),SpatialTransformer(640,8,80)]) for i in range(3)
            ],
            [
                Upsample(640,640)
            ],
            [
                nn.ModuleList([Concatenate(),ResBlock(640*2,320),SpatialTransformer(320,8,40)]) for i in range(3)
            ]
        ]

        self.exit_flow = nn.Sequential(            
                nn.GroupNorm(32,320),
                Activation("swish"),
                PaddedConv2D(320,4,kernel_size=3)            
        )



    def forward(self,inputs):
        latent, context, t_emb = inputs

        t_emb = self.time_embedding(t_emb)

        outputs = []

        x = self.entry_conv(latent)
        outputs.append(x)

        for i in range(len(self.downblocks)):            
            if(len(self.downblocks[i]) == 1):
                x = self.downblocks[i][0](x)
                outputs.append(x)
            else:
                for j in range(len(self.downblocks[i])): 
                    for layer in self.downblocks[i][j]:
                        x = layer([x,t_emb,context])
                    outputs.append(x)

        x = self.middleblocks([x,t_emb,context])

        for i in range(len(self.upblocks)):            
            if(len(self.upblocks[i]) == 1):
                x = self.upblocks[i][0](x)                
            else:
                for j in range(len(self.upblocks[i])):
                    x = [x,outputs.pop()]
                    for layer in self.downblocks[i][j]:
                        x = layer([x,t_emb,context])                      
        
        x = self.exit_flow(x)

        return x

class Concatenate(nn.Module):
    """Some Information about Concatenate"""
    def __init__(self):
        super(Concatenate, self).__init__()

    def forward(self, x):
        x = torch.concat(x,dim=-1)
        return x

class TimeEmbedding(nn.Module):
    """Some Information about TimeEmbedding"""
    def __init__(self,in_features=320,out_features=1280):
        super(TimeEmbedding, self).__init__()
        self.proj1 = nn.Linear(in_features,out_features)
        self.activation = Activation("swish")
        self.proj2 = nn.Linear(out_features,out_features)

    def forward(self, x):
        x = self.proj1(x)
        x = self.activation(x)
        x = self.proj2(x)
        return x

class GEGLU(nn.Module):
    """Some Information about GEGLU"""
    def __init__(self,in_features, out_features):
        super(GEGLU, self).__init__()
        self.out_features = out_features
        self.linear = nn.Linear(in_features,out_features)

    def forward(self, x):
        x = self.linear(x)
        x,gate = x[...,: self.out_features],x[...,self.out_features:]
        tanh_res = torch.tanh(
            gate * 0.7978845608 * (1 + 0.044715 * (gate**2))
        )

        return x * 0.5 * gate * (1 + tanh_res)

class Upsample(nn.Module):
    """Some Information about Upsample"""
    def __init__(self,in_channels,out_channels,kernel_size=3):
        super(Upsample, self).__init__()
        self.upsample = nn.Upsample((2,2))
        self.conv = PaddedConv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size
        )


    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)

        return x
    
class CrossAttention(nn.Module):
    """Some Information about CrossAttention"""
    def __init__(self,in_features,num_heads,head_size):
        super(CrossAttention, self).__init__()
        out_features = num_heads * head_size
        self.to_q = nn.Linear(in_features,out_features)
        self.to_k = nn.Linear(in_features,out_features)
        self.to_v = nn.Linear(in_features,out_features)

        self.scale = head_size**-0.5

        self.num_heads = num_heads
        self.head_size = head_size
        self.out_proj = nn.Linear(in_features,out_features)
        self.activation = Activation("softmax")

    def forward(self, inputs, context=None):
        if context is None:
            context = inputs
        q, k, v = self.to_q(inputs), self.to_k(context), self.to_v(context)
        q = torch.reshape(
            q, (-1, inputs.shape[1], self.num_heads, self.head_size)
        )
        k = torch.reshape(
            k, (-1, context.shape[1], self.num_heads, self.head_size)
        )
        v = torch.reshape(
            v, (-1, context.shape[1], self.num_heads, self.head_size)
        )

        q = torch.permute(q, (0, 2, 1, 3))  # (bs, num_heads, time, head_size)
        k = torch.permute(k, (0, 2, 3, 1))  # (bs, num_heads, head_size, time)
        v = torch.permute(v, (0, 2, 1, 3))  # (bs, num_heads, time, head_size)

        score = q.matmul(k.T) * self.scale
        weights = self.activation(
            score
        )  # (bs, num_heads, time, time)
        attn = weights.matmul(v.T)
        attn = torch.permute(
            attn, (0, 2, 1, 3)
        )  # (bs, time, num_heads, head_size)
        out = torch.reshape(
            attn, (-1, inputs.shape[1], self.num_heads * self.head_size)
        )
        return self.out_proj(out)

class BasicTransformerBlock(nn.Module):
    """Some Information about BasicTransformerBlock"""
    def __init__(self,in_features,out_features,num_heads,head_size):
        super(BasicTransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(in_features)
        self.attn1 = CrossAttention(in_features,num_heads, head_size)
        self.norm2 = nn.LayerNorm(in_features)
        self.attn2 = CrossAttention(in_features,num_heads, head_size)
        self.norm3 = nn.LayerNorm(in_features)
        self.geglu = GEGLU(in_features, out_features * 4)
        self.dense = nn.Linear(out_features*4, out_features)

    def forward(self, inputs):
        inputs, context = inputs
        x = self.attn1(self.norm1(inputs), context=None) + inputs
        x = self.attn2(self.norm2(x), context=context) + x
        return self.dense(self.geglu(self.norm3(x))) + x

class SpatialTransformer(nn.Module):
    """Some Information about SpatialTransformer"""
    def __init__(self,in_channels,num_heads,head_size):
        super(SpatialTransformer, self).__init__()
        self.norm = nn.GroupNorm(32,in_channels)
        out_channels = num_heads*head_size
        self.proj1 = PaddedConv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1
        )

        self.transformer_block = BasicTransformerBlock(
            in_channels,out_channels,num_heads,head_size
        )

        self.proj2 = PaddedConv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1
        )

    def forward(self, inputs):

        inputs,t_emb, context = inputs
        b,c, h, w = inputs.shape
        x = self.norm(inputs)
        x = self.proj1(x)
        x = torch.reshape(x, (b, h * w, c))
        x = self.transformer_block([x, context])
        x = torch.reshape(x, (b, c,h, w))

        return self.proj2(x) + inputs

class ResBlock(nn.Module):
    """Some Information about ResBlock"""
    def __init__(self,in_channels,out_channels,t_emb_in_features=1280,kernel_size=3,use_dws=False):
        super(ResBlock, self).__init__()
        self.entry_flow = [
            nn.GroupNorm(32,in_channels),
            Activation("swish"),
            PaddedConv2D(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                use_dws=use_dws
            )            
        ]

        self.embedding_flow = [
            Activation("swish"),
            nn.Linear(t_emb_in_features,out_channels)            
        ]

        self.exit_flow = [
            nn.GroupNorm(32,out_channels),
            Activation("swish"),
            PaddedConv2D(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                use_dws=use_dws
            )    
        ]

        if in_channels!=out_channels:
            self.residual_projection = PaddedConv2D(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                use_dws=use_dws
            )    
        else: 
            self.residual_projection = nn.Identity()

    def forward(self, x):

        inputs, embeddings, context = x
        x = inputs
        
        for layer in self.entry_flow:
            x = layer(x)
        for layer in self.embedding_flow:
            embeddings = layer(embeddings)
        x = x + embeddings.unsqueeze(-1).unsqueeze(-1)
        for layer in self.exit_flow:
            x = layer(x)
        return x + self.residual_projection(inputs)

class PaddedConv2D(nn.Module):
    """Some Information about PaddedConv2D"""
    def __init__(self,in_channels,out_channels,kernel_size=3,stride=1,use_dws=False,t=3):
        super(PaddedConv2D, self).__init__()

        if use_dws:
            self.conv = nn.ModuleList(
                [
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=in_channels*t,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding="same",
                        groups=in_channels
                    ),
                    nn.Conv2d(
                        in_channels=in_channels*t,
                        out_channels=out_channels,
                        kernel_size=1,                        
                        padding="same",
                    )
                ]
            )
        else: 
            self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=1
        )

    def forward(self, x):

        x = self.conv(x)
        return x
    
class Activation(nn.Module):
    """Some Information about Activation"""
    def __init__(self, activation_name="swish"):
        super(Activation, self).__init__()
        if activation_name == "swish":
            self.activation = nn.SiLU()
        elif activation_name == "softmax":
            self.activation = nn.Softmax(-1)
        else: 
            self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(x)
        return x