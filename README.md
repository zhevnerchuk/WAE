# WAE
Project for BMML course, Skoltech 2018

The goal of the project is to dive into Wasserstein Auto-Encoder (WAE) described by [Tolstikhin et al.](https://openreview.net/pdf?id=HkL7n1-0b)

They claim that WAEs have the same learning properties as VAEs (stable training, encoder-decoder architecture,
nice latent manifold structure) but generate samples of better quality.

Basic:
1. Discuss model in details
2. Reproduce experiments

Extensions:
1. Authors claim their approach can be used for any divergency, but experiments in the paper deal with only Jensen-Shannon one.
2. Try to apply WAEs to 3D point clouds generating

Team:
 - Ivan Barabanau
 - Albert Matveev 
 - Anton Zhevnerchuk
