## **Aim**

There are two aims for this project.
#### **Aim 1**

Reproduce **Tables 1 - 4** of *Yoon and Jung (2025)*. These tables numerically show that the **Theorem 7** is valid. **Tables 1, 2** are under multivariate normal data, which satisfies the assumption of **Theorem 7**. On the other hand, **Tables 3, 4** (in **Supplementary Information**) are under multivariate t data, which does not satisfy the assumption of **Theorem 7**. Moreover, as there is no plot in the paper regarding simulation studies, we will draw additional appropriate plots using the simulation results.
###### Theorem 7 (Yoon and Jung, 2025)
Under mild assumptions, as $p \rightarrow \infty$, $\mathbb{P}\big(\theta_{k}(\hat{\mathcal{U}}_m^{ARG}, \mathcal{U}_m) < \theta_{k}(\hat{\mathcal{U}}_m, \mathcal{U}_m)\big) \rightarrow 1$ for $k = 1,\dots,m$.

#### **Aim 2**

Figure out the convergence rate of **Theorem 4**, which plays an essential role in constructing the ARG estimator. Specifically, this project attempts to estimate $|u_i^{\top}d_j| = O_p(p^{-\alpha})$ by observing the plot of $p^{\alpha}|u_i^{\top}d_j|$ for various values of $\alpha$ ($u_i$ is the $j$th true PC direction, and $d_j$ will be defined in **Methods**). Note that the HDLSS asymptotics consider the sample size $n$ as a fixed value. This is beyond the contents of the paper, as **Theorem 4** only tells that $|u_i^{\top}d_j| = o_p(1)$.
###### Theorem 4 (Yoon and Jung, 2025)
Under mild assumptions, as $p \rightarrow \infty$, $\text{Angle}(u_i,d_j) \xrightarrow {P} \pi/2$ for $i=1,\dots,m$ and $j = 1,\dots,r$.


## Data-Generating Mechanisms

Although the paper imposes milder assumptions, the simulations in the paper are based on Gaussian distribution. Therefore, we use Gaussian setting for both **Aim 1 and 2**. Denote $e_1,e_2,e_3,e_4$ as follows: $$    \begin{split}
    e_1 = \frac{1}{\sqrt{p}}\begin{pmatrix}
        1_{p/4}\\1_{p/4}\\1_{p/4}\\1_{p/4}
    \end{pmatrix},
    e_2 = \frac{1}{\sqrt{p}}\begin{pmatrix}
        1_{p/4}\\1_{p/4}\\-1_{p/4}\\-1_{p/4}
    \end{pmatrix},
    e_3 = \frac{1}{\sqrt{p}}\begin{pmatrix}
        1_{p/4}\\-1_{p/4}\\-1_{p/4}\\1_{p/4}
    \end{pmatrix},
    e_4 = \frac{1}{\sqrt{p}}\begin{pmatrix}
        1_{p/4}\\-1_{p/4}\\1_{p/4}\\-1_{p/4}
    \end{pmatrix}.
    \end{split}$$

#### DGP for Aim 1 - For reproducing Table 1 and 3

Generate IID data $X_1,\dots X_n$ from a multivariate Gaussian (Table 1) / $t$ (Table 3) distribution $N(0_p,pe_1e_1^{\top} + 40 I_p)$ so that the true PC direction $u_1 = e_1$. Fix sample size $n = 40$. Set reference vector $v_1 = a_1 e_1 + \sqrt{1 - a_1^2} e_2$. Vary $\{a_1^2  = 0, 0.25, 0.5, 0.75, 1\}$ and $p = \{100, 200, 500, 1000, 2000\}$.

#### DGP for Aim 1 - For reproducing Table 2 and 4

Generate IID data $X_1,\dots,X_n$ from a multivariate Gaussian (Table 2) / $t$ (Table 4) distribution $N(0_p, 2pe_1e_1^{\top} + pe_2e_2^{\top} + 40 I_p)$ so that the true PC directions $u_1 = e_1$ and $u_2 = e_2$. Fix sample size $n = 40$. Set reference vectors $v_1 = e_1/2 + e_2/2 + e_3/2 + e_4/2$ and $v_2 = e_1/\sqrt{2} - e_3/\sqrt{2}$. Vary $p = \{100, 200, 500, 1000, 2000\}$.

#### DGP for Aim 2 - For figuring out convergence rate of Theorem 4

Generate IID data $X_1,\dots X_n$ from a multivariate Gaussian distribution $N(0_p,pe_1e_1^{\top} + I_p/SNR)$ so that the true PC direction $u_1 = e_1$. Fix sample size $n = 40$. Set reference vector $v_1 = 1/\sqrt{2} e_1 + 1/\sqrt{2} e_2$. Vary $SNR = \{0.01, 0.025, 0.05, 0.1, 0.25, 0.5\}$ and $p = \{100, 200, 500, 1000, 2000\}$.


## Estimands/Targets

The quantity used to evaluate the performance of the ARG estimator for **Aim 1** is the principal angle between two subspaces. The quantity used for **Aim 2** is the absolute value of the inner product between true PC direction $u_i$ and the negatively ridged discriminant vector $d_j$.


## Methods

The performance of *Adaptive Reference-Guided (ARG) estimator* for principal component subspace is compared to that of the naive estimator based on PCA. The goal of both estimators is to approximate PC subspace under $m$-spike covariance structure. That is, $m$ largest eigenvalues of covariance matrix is significantly larger (in terms of $p$) than the rest.

The definition of ARG estimator is as follows:
$$\hat{\mathcal{U}}_m^{ARG} := \mathcal{S} \setminus \mathcal{D}_r,$$
where $\mathcal{S} := \text{span}(\hat{u}_1,\dots,\hat{u}_m,v_1,\dots,v_r)$, $\mathcal{D} := \text{span} (d_1,\dots,d_r)$, $(\hat{u}_i,\hat{\lambda}_i)$ is the $i$th sample PC direction and variance obtained by the naive PCA, $v_j$ is the $j$th reference vector which are assumed to contain prior information about the true PC subspace, $d_j := -\tilde{\lambda}(S_m - \tilde{\lambda}I_p)^{-1}v_j$, $S_m := \sum_{i=1}^m \hat{\lambda}_i \hat{u}_i \hat{u}_i^{\top}$, and $\tilde{\lambda} := \sum_{i=m+1}^{n-1}\hat{\lambda}_i / (n-m-1)$. Note that the naive estimator based on PCA is $\hat{\mathcal{U}}_m = \text{span}(\hat{u}_1,\dots,\hat{u}_m)$.


## Performance Measures

For **Aim 1**, the paper just compares the principal angles between two subspaces for the ARG estimator and the naive estimator.

For **Aim 2**, there is no explicit quantitative measure of performance, as the primary goal is theoretical rather than comparative.