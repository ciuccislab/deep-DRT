# Project
## deep-DRT model to deconvolves the DRT from multidimensional EIS data


This repository contains some of the source code used for the paper titled *The deep-DRT: A deep neural network approach to deconvolve the distribution of relaxation times from multidimensional electrochemical impedance spectroscopy data*. Electrochimica Acta, 413, 140119. https://doi.org/10.1016/j.electacta.2021.139010. The article is available online at [Link](https://doi.org/10.1016/j.electacta.2021.139010) and in the [docs](docs) folder.

# Introduction
Electrochemical impedance spectroscopy (EIS) is a widely used characterization method in electrochemistry to gain
physical and chemical insights. Despite its power, the interpretation of EIS data is challenging. Physical models are usually problem-specific and equivalent circuits are not unique.
The distribution of relaxation times (DRT) approach overcomes these issues by assuming that electrochemical processes are relaxations characterized by their timescales [1]. However, the DRT deconvolution from experimental data is challenging because it is an ill-posed inverse problem and very sensitive to experimental errors [2,3]. To solve this issue many approaches have been used [2-5].

Moreover, while EIS spectra are collected at specific experimental conditions, they are typically deconvolved only relative to frequency. The deep-DRT model relaxes this limitation and considers EIS spectra to be a multidimensional function of the frequency and “experimental state” described by a vector of state variables (e.g., temperature and oxygen partial pressure).
Generalizing our previous model [6], the deep-DRT framework leverages deep neural networks to perform the DRT deconvolution.  Performing simulations with both articial and real experiments confirmed the capability of the deep-DRT model to perform both regression and prediction at untested conditions..

![image](https://user-images.githubusercontent.com/123150335/213650018-5eea45f5-94f1-4c42-946e-30bb84866953.png)

# Dependencies
numpy

scipy

matplotlib

pandas

torch

# Tutorials
1. **Ex1_ZARC.ipynb**: this notebook shows that the deep-DRT can catch the linear dependency of the charge transfer resistance on the scalar state variable $\psi$. The frequency range is from $10^{-4}$ Hz to $10^{4}$ Hz with 10 points per decade (ppd).
2. **Ex2_double_ZARC.ipynb** : this notebook shows that the deep-DRT can catch the linear dependency of the charge transfer resistance and the exponential dependence of the timescale constant on the scalar state variable $\psi$. The frequency range is from $10^{-4}$ Hz to $10^{4}$ Hz with 10 points per decade (ppd).
3. **Ex3_PWC.ipynb** : this notebook shows that the deep-DRT can catch the linear dependency of the charge transfer resistance on the scalar state variable $\psi$ in case the DRT shows discontinuities. The frequency range is from $10^{-4}$ Hz to $10^{4}$ Hz with 10 points per decade (ppd).
4. **Ex4_BLF_D5.ipynb** : this notebook shows an example with experimental data. The dataset is read from a folder containing EIS spectra saved in csv files. The DRT is deconvolved from 3D EIS data as functions of the temperature and partial pressure of oxygen.

# Citation

```
@article{Quattrocchi2019deepDRT,
  title={The deep-DRT: A deep neural network approach to deconvolve the distribution of relaxation times from multidimensional electrochemical impedance spectroscopy data},
  author={Quattrocchi, Wan, Belotti, Kim, Pepe, Ahmadi, and Ciucci, Francesco},
  journal={Electrochimica Acta},
  pages={139010},
  year={2021},
  publisher={Elsevier}
}

```

# References
[1] Ciucci, F. (2018). Modeling electrochemical impedance spectroscopy. Current Opinion in Electrochemistry.132-139 https://doi.org/10.1016/j.coelec.2018.12.003

[2] Wan, T. H., Saccoccio, M., Chen, C., & Ciucci, F. (2015). Influence of the discretization methods on the distribution of relaxation times deconvolution: implementing radial basis functions with DRTtools. Electrochimica Acta, 184, 483-499. https://doi.org/10.1016/j.electacta.2015.09.097

[3] Saccoccio, M., Wan, T. H., Chen, C., & Ciucci, F. (2014). Optimal regularization in distribution of relaxation times applied to electrochemical impedance spectroscopy: ridge and lasso regression methods-a theoretical and experimental study. Electrochimica Acta, 147, 470-482. https://doi.org/10.1016/j.electacta.2014.09.058

[4] Effat, M. B., & Ciucci, F. (2017). Bayesian and hierarchical Bayesian based regularization for deconvolving the distribution of relaxation times from electrochemical impedance spectroscopy data. Electrochimica Acta, 247, 1117-1129. https://doi.org/10.1016/j.electacta.2017.07.050

[5] Ciucci, F., & Chen, C. (2015). Analysis of electrochemical impedance spectroscopy data using the distribution of relaxation times: A Bayesian and hierarchical Bayesian approach. Electrochimica Acta, 167, 439-454. https://doi.org/10.1016/j.electacta.2015.03.123

[6] Liu, J., Ciucci, F., The Deep-Prior Distribution of Relaxation Times Journal of The Electrochemical Society, 167.2 (2020): 026506 https://doi.org/10.1149/1945-7111/ab631a
