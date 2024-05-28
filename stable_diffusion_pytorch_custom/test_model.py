from models.flash_fusion import FlashFusion
import torch 

model = FlashFusion()
print(model)


latent = torch.randn((1,4,64,64))
context = torch.randn((1,77,768))
t_emb = torch.randn((1,320))

model([latent,context,t_emb])
