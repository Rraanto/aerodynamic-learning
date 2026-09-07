# Summary

The search was fruitful for both requested goals. Multifidelity output combination is well covered by autoregressive Gaussian processes, nonlinear neural or Gaussian-process mappings, and basis-reduced methods for field-valued outputs. Low-to-high-fidelity calibration is also well covered by scale-plus-discrepancy models and learned residual corrections, including physics-informed variants. Aerodynamic spatial fields and time-dependent quantities have direct demonstrations, but evidence for methods validated on fully spatio-temporal aerodynamic fields remains comparatively thin; this is the clearest application gap.

# Reference methods

## Methods for combining multifidelity outputs

### Autoregressive co-kriging

Model each higher-fidelity response as a scaled lower-fidelity process plus an independent Gaussian-process discrepancy. This is the most established starting point when fidelities are nested, reasonably correlated, and uncertainty estimates matter; recursive and vector-valued extensions support multiple levels and random fields [@kennedy2000predicting; @parussini2017multifidelity].

### Nonlinear cross-fidelity learning

When the low-to-high relationship changes across the input domain or is strongly nonlinear, nonlinear autoregressive Gaussian processes and composite neural networks learn the cross-fidelity map rather than imposing a global linear scale. These methods are attractive for complex PDE response surfaces, though their flexibility raises data and validation requirements [@perdikaris2017nonlinear; @meng2020composite].

### Reduced-basis fusion for spatial and temporal outputs

For high-dimensional fields, first compressing outputs and then fusing fidelity levels in coefficient space avoids fitting a separate surrogate at every grid point. POD-co-kriging has been demonstrated on hypersonic surface-pressure distributions, while multifidelity recurrent networks directly target parameterized time-dependent outputs, including unsteady aerodynamic coefficients [@rokita2018multifidelity; @conti2023multifidelity].

## Methods for calibrating low-fidelity data with high-fidelity data

### Scale-plus-discrepancy calibration

Estimate a multiplicative scale and an additive discrepancy from paired high-fidelity observations, then correct the low-fidelity predictor throughout the input domain. The Bayesian formulation supplies predictive uncertainty and is the clearest baseline when the residual is smooth and high-fidelity samples are scarce [@kennedy2000predicting].

### Physics-informed discrepancy Gaussian processes

Physics-informed co-kriging builds the low-fidelity covariance from ensembles of physical-model outputs and learns a separate Gaussian-process discrepancy against high-fidelity observations. It is especially relevant when calibration should preserve known linear physical constraints while assimilating sparse field measurements [@yang2019physics].

### Learned residual and parameter calibration

Composite neural networks can learn linear and nonlinear low-to-high corrections, while corrected low-fidelity surrogates can be embedded in Bayesian parameter calibration to retain computational speed. The latter has a direct time-history dynamics demonstration near a hypersonic vehicle, but transferring it to aerodynamic flow fields requires application-specific validation [@meng2020composite; @absi2016multifidelity].
