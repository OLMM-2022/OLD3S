# Online Deep Learning from Doubly-Streaming Data
![Python 3.9](https://img.shields.io/badge/python-3.9-green.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
## Abstract
This paper investigates a new online learning problem with doubly-streaming data,
where the data streams are described by feature spaces that constantly evolve.
    The challenges of this problem are two folds.
    1) Data instances that flow in ceaselessly
    are not likely to always follow an identical distribution,
    require the learners to be updated on-the-fly.
    2) New features that just emerge are described by 
    very few data instances, 
    result in \emph{weak} learners that tend to make error predictions.
    
    To overcome,
    a plausible idea is to establish relationship
    between the pre-and-post evolving feature spaces,
    so that an online learner can leverage and adapt 
    the learned knowledge from the old 
    to the new features for better performance.
    
    Unfortunately, this idea does not scale up to 
    high-dimensional media streams 
    with complex feature interplay,
    suffering an tradeoff between onlineness 
    (biasing shallow learners)
    and expressiveness (requiring deep learners).

    Motivated by this,
    we propose a novel \myAlg\ paradigm,
    where a shared latent subspace is discovered 
    to  summarize information from the old and new feature spaces,
    building intermediate feature mapping relationship.

    
    A key trait of \myAlg\ is to treat
    the {\em model capacity} as a learnable semantics,
    yields optimal model depth and parameters jointly in accordance 
    with the complexity and non-linearity of the inputs
    in an online fashion.
    
    Both theoretical analyses and extensive experiments benchmarked on
    real-world datasets including images and natural languages
    substantiate the viability and effectiveness of our proposal.
## Requirements
This code was tested on Linux(Ubuntu) and macOS
```
conda create --name GAFNC python=3.7.10
pip install torch==1.6.0+cpu torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
pip install torch-scatter==2.0.6 -f https://data.pyg.org/whl/torch-1.6.0%2Bcpu.html
pip install torch-sparse==0.6.9 -f https://data.pyg.org/whl/torch-1.6.0%2Bcpu.html
pip install torch-geometric==1.7.0
pip install torch-cluster==1.5.9 -f https://data.pyg.org/whl/torch-1.6.0%2Bcpu.html
pip install torch-spline-conv==1.2.1 -f https://data.pyg.org/whl/torch-1.6.0%2Bcpu.html
conda install -c conda-forge pyod
```

## Run
```angular2html
conda activate GAFNC
zsh run_global_attack_on_Cora.sh
zsh run_target_attack_on_Cora.sh
```

