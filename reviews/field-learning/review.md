# Summary

The search was fruitful for general functional dimension reduction and clustering, including methods designed for multivariate functions and two-dimensional surfaces. These methods transfer naturally to treating each airfoil as a vector-valued field over angle of attack and Reynolds number. Direct aerodynamic coverage is weaker: verified studies embed pressure fields or map scalar performance objectives, but none found clusters many airfoils from complete aerodynamic-performance surfaces over both operating variables. The main methodological caution is that a two-dimensional display is not necessarily a reliable clustering space; clusters should normally be fitted and validated in a richer functional representation, then shown in two dimensions.

# Reference methods

## Methods for clustering high dimensional parameterised airfoils into 2D points

### Multivariate functional PCA followed by clustering

Smooth each aerodynamic response surface, combine outputs such as lift and drag as multivariate functional data, and represent every airfoil by functional principal-component scores. The first two scores provide an interpretable linear map, while clustering can use enough retained scores to preserve relevant variation; Gaussian-mixture clustering of multivariate functional scores supplies probabilistic memberships [@happ2018multivariate; @jacques2014model].

### Cluster-aware functional subspaces

Functional factorial k-means estimates the grouping and the low-dimensional subspace together, avoiding the risk that variance-maximizing principal components omit directions that separate airfoils. It is a useful alternative when the primary objective is cluster discrimination rather than reconstruction, although a two-dimensional restriction should be checked against higher-dimensional solutions [@yamamoto2014functional].

### Tensor-product surface clustering

Represent each angle-of-attack--Reynolds surface in a tensor-product spline basis, then apply k-means or spectral clustering to the coefficient representation. This method directly handles bivariate surfaces and measurement error, making it the most literal statistical match to the proposed aerodynamic-performance field [@zambom2022basis].

### Nonlinear functional manifold embeddings

When stall and regime transitions create nonlinear variation, functional Isomap-style coordinates or functional diffusion maps can produce two-dimensional embeddings from distances between functions. They preserve manifold or multiscale neighborhood structure better than linear FPCA in suitable cases, but cluster stability should be assessed independently of the plotted coordinates [@chen2012nonlinear; @barroso2024functional].

## Airfoil classification methods based on their aerodynamic performance on 2D parameter space

### Operating-region-weighted functional clustering

Sparse functional clustering learns which parts of a domain distinguish groups instead of weighting all operating conditions equally. Extending its weight function from curves to the angle-of-attack--Reynolds surface is a transferable idea, not a directly validated airfoil method; tensor-product surface clustering supplies the required bivariate representation [@floriello2017sparse; @zambom2022basis].

### Self-organizing performance maps

Self-organizing maps give a topology-preserving two-dimensional lattice that can expose performance/design trade-offs and cluster neighborhoods. A direct airfoil study maps CFD-derived maximum lift and maximum lift-to-drag objectives together with PARSEC variables, but it reduces each airfoil to scalar objectives rather than using its full operating-domain field [@jung2016implementation].

### Aerodynamic-field latent representations

Nonlinear embeddings and convolutional autoencoders have been applied directly to airfoil pressure fields: one compares kernel PCA, locally linear embedding, and t-SNE for transient CFD snapshots, while another learns a very low-dimensional representation that separates linear and stalled aerodynamic behavior. These results support nonlinear performance embeddings, but they address flow states or stall prediction rather than clustering multiple airfoils across a two-parameter operating domain [@vasudevan2023manifold; @saetta2022machine].

# Essential reviews and state of the art

## General reviews

### Functional data analysis

Wang, Chiou, and Müller provide a compact overview of FPCA, nonlinear dimension reduction, clustering, and classification. It is the best broad entry point for formulating aerodynamic response surfaces as functional observations [@wang2016functional].

### Model-based functional clustering and classification

Chamroukhi and Nguyen organize mixture-based methods for curves and surfaces, including heterogeneous, incomplete, and dynamically structured data. It is the most focused navigation anchor for probabilistic functional clustering [@chamroukhi2019model].

## Application-specific reviews

### Machine learning in aerodynamic shape optimization

Li, Du, and Martins review compact design spaces, aerodynamic surrogates, and optimization architectures for airfoils and wings. It gives the relevant aerodynamic context and clarifies that most dimensionality-reduction work targets geometry or flow surrogates rather than performance-surface classification [@li2022machine].
