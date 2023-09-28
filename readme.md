## mainDCGAN.py

- generator: DCGAN
- main nets: CNNs

parameters (arg)

- bs: batch size for test
- local_bs: batch size of local real samples

- models
- dataset

- wu_epochs
- gen_wu_epochs
- epochs
- local_ep_gen
- gen_local_ep

- aid_by_gen

- args.img_shape (1, 16, 16)

## Generator
- Conv-based generator forward: img_ch, img_size, img_size
- MLP-based generator forward: img_ch x img_size x img_size

- sample_image(self, args) returns [gen_imgs, one_c]
    - generates random noises and labels of args.local_bs
    - sample_image_4visualization returns gen_imgs


## LocalUpdate
to perform local training of main nets
output trained weights of a main net

if generator exists,
- batch iteration: len(local dataset)/local_bs 
- gennet.sample_image: geneartes local_bs samples
- network starts from "feature part"

then, train with local samples
a network consists of feature extractor and the rest

## LocalUpdate_DCGAN
to perform local training of a generator
output trained weights of a generator and a discriminator


## cgan_fed.py / cvae_fed.py / cddpm_fed.py
train CGAN/CVAE by federated learning


## trainbyGen.py
train main net by real-time generated samples (+ local samples)
> Note that CDDPM takes too much time.
> For evaluation of DDPM, we generate samples before training by ``imageGeneration.py''
> and train main net by generated samples (+ local samples) by ``trainbyGenedSamples.py''


## trainFeatureNet.py
train common feature extractor
it is implemented only by a single net, i.e., feature extractor + the rest
(to be) how about CNN-based nets ?


## cgan_feat_fed.py & cvae_feat_fed.py & cddpm_feat_fed.py (to be)
load trained feature extractor from trainFeatureNet.py
and train generators that generate output images from the trained feature extractor
(to be) DDPM with less n_T -> will it be still effective ?


## DIRECTORY modelsMNIST
generators that generate raw samples (MNIST 28x28)


## DIRECTORY models
generators that generate features (e.g., MNIST 14x14 or 16x16)