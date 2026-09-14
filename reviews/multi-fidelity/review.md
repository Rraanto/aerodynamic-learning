# Summary

The search was fruitful for both goals. Gaussian-process, reduced-basis, and neural/operator methods provide strong coverage for scalar, vector, spatial-field, and time-dependent outputs. Aerodynamic evidence includes lift/drag databases, hypersonic pressure distributions, unsteady loads, and one direct spatio-temporal airfoil-flow study. The weakest area remains independent validation on fully spatio-temporal aerodynamic fields with realistic mesh mismatch, noisy measurements, and changing flow regimes; method comparisons are mostly case-specific.

# Reference methods

## Methods for combining multifidelity outputs

### Autoregressive Gaussian processes and co-kriging

Represent the high-fidelity response as a scaled low-fidelity Gaussian process plus an independent discrepancy. This is the clearest probabilistic baseline when fidelities are correlated and high-fidelity samples are scarce; nonlinear autoregressive GPs replace the global linear link when correlation varies over the input domain [@kennedy2000predicting; @perdikaris2017nonlinear].

### Multi-output and reduced-basis field fusion

For field-valued outputs, fuse coefficients in a spectral or POD basis rather than fitting every grid value independently. Multivariate recursive GPs retain cross-output covariance, while POD-co-kriging has a direct demonstration on hypersonic surface-pressure distributions and permits different low- and high-fidelity output dimensions [@parussini2017multi; @rokita2018multifidelity].

### Spatio-temporal Gaussian-process fusion

Embed spatial coordinates, time, low-fidelity fields, and selected derivatives in a nonlinear Gaussian-process map. This directly targets sparse high-fidelity measurements combined with spatially and temporally dense simulations; it has been tested with RANS, LES, and DNS data around a NACA 0012 airfoil. Hierarchical kriging with time-delayed inputs is a complementary choice for unsteady aerodynamic load histories and supplies predictive uncertainty [@zhang2022fusion; @peng2023multi].

### Multifidelity neural operators

Learn the low-to-high map between functions rather than only between fixed-dimensional vectors. Composite multifidelity DeepONets support nonlinear field mappings and physics-informed training; a newer architecture adds physics-guided subsampling for data-efficient spatio-temporal unsteady-flow prediction [@howard2023multifidelity; @yang2025data].

## Methods for calibrating low-fidelity data with high-fidelity data

### Scale-plus-discrepancy calibration

Estimate a scale and an additive discrepancy from paired fidelities, then propagate the corrected low-fidelity predictor with uncertainty. The original autoregressive construction is a suitable baseline for simulator-to-simulator correction, while the multifidelity calibration extension jointly incorporates field observations and simulator-specific calibration parameters [@kennedy2000predicting; @goh2013prediction].

### Physics-informed discrepancy Gaussian processes

Construct the low-fidelity covariance from ensembles of physical-model outputs and learn a separate Gaussian-process discrepancy from sparse high-fidelity observations. This preserves deterministic linear physical constraints up to a derived error bound and is useful when calibration should not discard known model structure [@yang2019physics].

### Learned nonlinear correction maps

Composite neural networks learn linear and nonlinear correction components jointly and scale better than exact Gaussian processes as sample size and output dimension grow. The general construction covers inverse PDE problems, and an aerodynamic specialization corrects Euler-derived lift and drag using sparse Navier--Stokes results [@meng2020composite; @he2020multi].

# Essential reviews and state of the art

## General reviews

### Multifidelity methods for outer-loop computation

Peherstorfer, Willcox, and Gunzburger organize methods into adaptation, fusion, and filtering and connect them to uncertainty propagation, inference, and optimization. It is the strongest foundational navigation source for deciding how a surrogate fits into a larger computational workflow [@peherstorfer2018survey].

### Modern multifidelity-model taxonomy

Fernández-Godino surveys fidelity definitions, combination strategies, sampling, reproducibility, benchmarking, and cost reporting across engineering applications. It complements the older survey with a more current and surrogate-focused taxonomy [@fernandezgodino2023review].

## Application-specific reviews

### Data-driven unsteady aerodynamics and aeroelasticity

Kou and Zhang review system identification, feature extraction, and data fusion for aerodynamic fields and loads. It is the most useful application anchor for placing multifidelity fusion within unsteady aerodynamic reduced-order modeling [@kou2021data].
