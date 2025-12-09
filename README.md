# Tip Box

Collection of simple and box models for tipping point analysis

## Developers Guide

1. Create conda environment
```
conda create -n tipbox python=3.10
conda activate tipbox
```

2. Install package
```
git clone https://github.com/juannat7/tipbox
cd tipbox
pip install -e .
```


> `AUTO-07P` software is required for bifurcation-induced data generation. See `docs/INSTALL_AUTO.md` for detailed instructions.

## Future Work

- [ ] Add Rozenzweig−MacArthur consumer−resource model (Transcritical) -- Bury 2021
- [ ] Add SEIRx disease dynamics and population vaccinating behavior model (Transcritical) -- Bury 2021
- [ ] Add three-species chaotic food chain model (Hopf) -- Zhuge 2025
- [ ] Add Plant-herbivore (Rate) -- Ritchie 2023
- [ ] Add AMOC 5 box models (Rate) -- Ritchie 2023