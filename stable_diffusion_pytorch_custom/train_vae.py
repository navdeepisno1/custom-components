# https://github.com/aandyw/diffusers/blob/55337fc2d320485ecd8bccd48ae23c1bf68e23eb/examples/vae/train_vae.py#L424

import tensorflow as tf 
import keras 
from models.flash_vae.flash_vae import FlashEncoder,FlashDecoder
import os 
import numpy as np 

class Args:
    def __init__(self):
        self.learning_rate = 4e-4
        self.batch_size = 16
        self.latent_channels = 16
        self.downsampling_factor = 8 
        self.epochs = 100
        self.save_dir = "saved_models"
        self.data_dir = ""
        self.expt_name = ""
        self.image_shape = 512 
        self.use_mixed_precision = False
        self.lpips_scale_factor = 1.0
        self.kl_scale_factor = 1.0

args = Args()

def get_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.io.decode_png(image, 3)
    image = tf.image.resize(image, (512, 512))
    return image 

def get_dataset(path,batch_size):
    image_paths = [os.path.join(path,i) for i in os.listdir(path)]

    dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    dataset = dataset.map(get_image)
    dataset = dataset.batch(batch_size)
    return dataset



def get_encoder():
    x = keras.layers.Input((512,512,3))
    y = FlashEncoder()(x)

    model = keras.Model(x,y)
    return model  

def get_decoder():
    x = keras.layers.Input((64,64,16))
    y = FlashDecoder()(x)

    model = keras.Model(x,y)
    return model

def get_inter_block():
    x = keras.layers.Input((None,None,args.latent_channels))
    y = keras.layers.Conv2D(args.latent_channels*2,1,padding="same")(x)
    y = keras.layers.Conv2D(args.latent_channels,1,padding="same")(y)
    model = keras.Model(x,y)

class VaeTrainer(tf.keras.Model):
    def __init__(self,encoder,decoder,inter_block,**kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.inter_block = inter_block

        self.encoder.trainable = True
        self.decoder.trainable = True
        self.inter_block.trainable = True
    
    def compute_loss(self,target,predictions):
        loss_fn = tf.keras.losses.MeanSquaredError()
        loss = loss_fn(target,predictions)
        return loss 
    

    def get_mean(self,x):
        mean, logvar = tf.split(x, 2, axis=-1)
        return mean 

    def train_step(self,inputs):
        with tf.GradientTape() as tape:
            target = inputs
            latents = self.encoder(inputs)            
            z = self.get_mean(latents)
            z = self.inter_block(z)
            predictions = self.decoder(z)
            loss = self.compute_loss(target,predictions)
        
        trainable_vars = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(loss,trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        return {
            "loss": loss 
        }

trainer = VaeTrainer(
    encoder = get_encoder(),
    decoder = get_decoder()
)

trainer.compile(optimizer="adam")

dataset = get_dataset(args.data_dir,args.batch_size)

trainer.fit(dataset,epochs=args.epochs)
