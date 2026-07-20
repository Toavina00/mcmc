# AIMS Research Project

**Project title**: Riemannian Manifold Hamiltonian Monte Carlo and Low-Rank Approximation for Gaussian Process Regression


## Implemented samplers

JAX implementation of Markov Chain Monte Carlo (MCMC) sampling methods:
- **Metropolis-Hastings (MH)** — `mcmc.mh.sample`
- **Hamiltonian Monte Carlo (HMC)** — `mcmc.hmc.sample`
- **Riemannian Manifold HMC (RMHMC)** — `mcmc.rmhmc.sample`
- **Rank-1 RMHMC (R1-RMHMC)** — `mcmc.r1_rmhmc.sample`

## Project layout

```
.
├── src/mcmc/        # library code
│   ├── mh.py
│   ├── hmc.py
│   ├── rmhmc.py
│   ├── r1_rmhmc.py
│   └── utils.py     # MCMC diagnostics utilities
├── thesis/          # thesis manuscript
├── pyproject.toml
└── README.md
```

## References

1. Baydin, A. G., Pearlmutter, B. A., Radul, A. A., & Siskind, J. M. (2018).
   Automatic differentiation in machine learning: a survey. *Journal of Machine
   Learning Research*, 18(153), 1–43. <http://jmlr.org/papers/v18/17-468.html>
2. Box, G. E. P., & Muller, M. E. (1958). A note on the generation of random normal
   deviates. *Annals of Mathematical Statistics*, 29, 610–611.
3. Bradbury, J., Frostig, R., Hawkins, P., Johnson, M. J., Katariya, Y., Leary, C.,
   Maclaurin, D., Necula, G., Paszke, A., VanderPlas, J., Wanderman-Milne, S., &
   Zhang, Q. (2018). *JAX: composable transformations of Python+NumPy programs*.
   <https://github.com/jax-ml/jax>
4. Brookes, M. (2020). *The matrix reference manual*.
   <http://www.ee.imperial.ac.uk/hp/staff/dmb/matrix/intro.html>
5. Brooks, S., Gelman, A., Jones, G., & Meng, X.-L. (2011). *Handbook of Markov
   Chain Monte Carlo*. <https://doi.org/10.1201/b10905>
6. Carpenter, B., Gelman, A., Hoffman, M. D., Lee, D., Goodrich, B., Betancourt, M.,
   Brubaker, M., Guo, J., Li, P., & Riddell, A. (2017). Stan: A probabilistic
   programming language. *Journal of Statistical Software*, 76(1).
7. Eckart, C., & Young, G. (1936). The approximation of one matrix by another of
   lower rank. *Psychometrika*, 1(3), 211–218. <https://doi.org/10.1007/BF02288367>
8. Geyer, C. J. (1992). Practical Markov chain Monte Carlo. *Statistical Science*,
   7(4), 473–483. <http://www.jstor.org/stable/2246094>
9. Girolami, M., & Calderhead, B. (2011). Riemann manifold Langevin and Hamiltonian
   Monte Carlo methods. *Journal of the Royal Statistical Society Series B:
   Statistical Methodology*, 73(2), 123–214. <https://doi.org/10.1111/j.1467-9868.2010.00765.x>
10. Golub, G. H., & Van Loan, C. F. (2013). *Matrix Computations* (4th ed.). Johns
    Hopkins University Press. <https://doi.org/10.1137/1.9781421407944>
11. Hastings, W. K. (1970). Monte Carlo sampling methods using Markov chains and
    their applications. *Biometrika*, 57(1), 97–109. <https://doi.org/10.1093/biomet/57.1.97>
12. Hayakawa, T., & Asai, S. (2025). *Fast Riemannian-manifold Hamiltonian Monte
    Carlo for hierarchical Gaussian-process models*. <https://arxiv.org/abs/2511.06407>
13. Hoffman, M. D., & Gelman, A. (2011). *The No-U-Turn sampler: Adaptively setting
    path lengths in Hamiltonian Monte Carlo*. <https://arxiv.org/abs/1111.4246>
14. MacKay, D. J. C. (2003). *Information theory, inference and learning algorithms*.
    Cambridge University Press.
15. Metropolis, N., Rosenbluth, A. W., Rosenbluth, M. N., Teller, A. H., & Teller, E.
    (1953). Equation of state calculations by fast computing machines. *The Journal
    of Chemical Physics*, 21(6), 1087–1092. <https://doi.org/10.1063/1.1699114>
16. Neal, R. M. (2012). *MCMC using Hamiltonian dynamics*. <https://arxiv.org/abs/1206.1901>
17. Paquet, U., & Fraccaro, M. (2018). *An efficient implementation of Riemannian
    manifold Hamiltonian Monte Carlo for Gaussian process models*. <https://arxiv.org/abs/1810.11893>
18. Pearlmutter, B. A. (1994). Fast exact multiplication by the Hessian. *Neural
    Computation*, 6(1), 147–160. <https://doi.org/10.1162/neco.1994.6.1.147>
19. Phan, D., Pradhan, N., & Jankowiak, M. (2019). *Composable effects for flexible
    and accelerated probabilistic programming in NumPyro*. <https://arxiv.org/abs/1912.11554>
20. Press, W. (2007). *Numerical Recipes 3rd Edition: The Art of Scientific Computing*.
    Cambridge University Press. <https://numerical.recipes/book.html>
21. Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian processes for machine
    learning*. The MIT Press.
22. Robert, C. P., & Casella, G. (2004). *Monte Carlo statistical methods* (Springer
    Texts in Statistics).
23. Williams, C., & Seeger, M. (2001). Using the Nyström method to speed up kernel
    machines. In T. Leen, T. Dietterich, & V. Tresp (Eds.), *Advances in Neural
    Information Processing Systems 13 (NIPS 2000)* (pp. 682–688). MIT Press.
