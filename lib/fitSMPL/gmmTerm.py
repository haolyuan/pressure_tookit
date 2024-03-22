import numpy as np
import os
import pickle
import torch
from typing import TypeVar

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
_dtype = TypeVar('_dtype')

# Ref: xrmocap
# https://github.com/openxrlab/xrmocap/blob/3a6b40397f9e3d36c87c11c6c8a3e435c7b0a094/xrmocap/model/loss/prior_loss.py#L326


class MaxMixturePriorLoss(torch.nn.Module):

    def __init__(self,
                 prior_folder: str = './data',
                 num_gaussians: int = 8,
                 dtype: _dtype = torch.float32,
                 epsilon: float = 1e-16,
                 use_merged: bool = True,
                 reduction: Literal['mean', 'sum', 'none'] = 'mean',
                 loss_weight: float = 1.0):
        """Ref: SMPLify-X
        https://github.com/vchoutas/smplify-x/blob/master/smplifyx/prior.py

        Args:
            prior_folder (str, optional):
                Path to the folder for prior file.
                Defaults to './data'.
            num_gaussians (int, optional):
                . Defaults to 8.
            dtype (_dtype, optional):
                Defaults to torch.float32.
            epsilon (float, optional):
                Defaults to 1e-16.
            use_merged (bool, optional):
                . Defaults to True.
            reduction (Literal['mean', 'sum', 'none'], optional):
                The method that reduces the loss to a
                scalar. Options are 'none', 'mean' and 'sum'.
                Defaults to 'mean'.
            loss_weight (float, optional):
                The weight of the loss. Defaults to 1.0.
        """
        super(MaxMixturePriorLoss, self).__init__()

        assert reduction in (None, 'none', 'mean', 'sum')
        self.reduction = reduction
        self.loss_weight = loss_weight

        if dtype == torch.float32:
            np_dtype = np.float32
        elif dtype == torch.float64:
            np_dtype = np.float64
        else:
            raise TypeError(f'Unknown float type {dtype}, exiting!')

        self.num_gaussians = num_gaussians
        self.epsilon = epsilon
        self.use_merged = use_merged
        gmm_fn = f'gmm_{num_gaussians:02d}.pkl'

        full_gmm_fn = os.path.join(prior_folder, gmm_fn)
        if not os.path.exists(full_gmm_fn):
            raise FileNotFoundError(
                f'The path to the mixture prior "{full_gmm_fn}"' +
                ' does not exist, exiting!')

        with open(full_gmm_fn, 'rb') as f:
            gmm = pickle.load(f, encoding='latin1')

        if type(gmm) == dict:
            means = gmm['means'].astype(np_dtype)
            covs = gmm['covars'].astype(np_dtype)
        elif 'sklearn.mixture.gmm.GMM' in str(type(gmm)):
            means = gmm.means_.astype(np_dtype)
            covs = gmm.covars_.astype(np_dtype)
        else:
            raise TypeError(
                f'Unknown type for the prior: {type(gmm)}, exiting!')

        self.register_buffer('means', torch.tensor(means, dtype=dtype))

        self.register_buffer('covs', torch.tensor(covs, dtype=dtype))

        precisions = [np.linalg.inv(cov) for cov in covs]
        precisions = np.stack(precisions).astype(np_dtype)

        self.register_buffer('precisions',
                             torch.tensor(precisions, dtype=dtype))

        # The constant term:
        sqrdets = np.array([(np.sqrt(np.linalg.det(c)))
                            for c in gmm['covars']])
        const = (2 * np.pi)**(69 / 2.)

        nll_weights = np.asarray(gmm['weights'] / (const *
                                                   (sqrdets / sqrdets.min())))
        nll_weights = torch.tensor(nll_weights, dtype=dtype).unsqueeze(dim=0)
        self.register_buffer('nll_weights', nll_weights)

        weights = torch.tensor(gmm['weights'], dtype=dtype).unsqueeze(dim=0)
        self.register_buffer('weights', weights)

        self.register_buffer('pi_term',
                             torch.log(torch.tensor(2 * np.pi, dtype=dtype)))

        cov_dets = [
            np.log(np.linalg.det(cov.astype(np_dtype)) + epsilon)
            for cov in covs
        ]
        self.register_buffer('cov_dets', torch.tensor(cov_dets, dtype=dtype))

        # The dimensionality of the random variable
        self.random_var_dim = self.means.shape[1]

    def get_mean(self) -> torch.Tensor:
        """Returns the mean of the mixture.

        Returns:
            torch.Tensor: mean of the mixture.
        """
        mean_pose = torch.matmul(self.weights, self.means)
        return mean_pose

    def merged_log_likelihood(self, pose):
        diff_from_mean = pose.unsqueeze(dim=1) - self.means

        prec_diff_prod = torch.einsum('mij,bmj->bmi',
                                      [self.precisions, diff_from_mean])
        diff_prec_quadratic = (prec_diff_prod * diff_from_mean).sum(dim=-1)

        curr_loglikelihood = 0.5 * diff_prec_quadratic - \
            torch.log(self.nll_weights)
        min_likelihood, _ = torch.min(curr_loglikelihood, dim=1)
        return min_likelihood

    def log_likelihood(self, pose: torch.Tensor) -> torch.Tensor:
        """Create graph operation for negative log-likelihood calculation.

        Args:
            pose (torch.Tensor):
                body_pose from smpl.

        Returns:
            torch.Tensor
        """
        likelihoods = []

        for idx in range(self.num_gaussians):
            mean = self.means[idx]
            prec = self.precisions[idx]
            cov = self.covs[idx]
            diff_from_mean = pose - mean

            curr_loglikelihood = torch.einsum('bj,ji->bi',
                                              [diff_from_mean, prec])
            curr_loglikelihood = torch.einsum(
                'bi,bi->b', [curr_loglikelihood, diff_from_mean])
            cov_term = torch.log(torch.det(cov) + self.epsilon)
            curr_loglikelihood += 0.5 * (
                cov_term + self.random_var_dim * self.pi_term)
            likelihoods.append(curr_loglikelihood)

        log_likelihoods = torch.stack(likelihoods, dim=1)
        min_idx = torch.argmin(log_likelihoods, dim=1)
        weight_component = self.nll_weights[:, min_idx]
        weight_component = -torch.log(weight_component)

        return weight_component + log_likelihoods[:, min_idx]

    def forward(
        self,
        body_pose: torch.Tensor,
        loss_weight_override: float = None,
        reduction_override: Literal['mean', 'sum',
                                    'none'] = None) -> torch.Tensor:
        """Forward function of MaxMixturePrior.

        Args:
            body_pose (torch.Tensor):
                The body pose parameters.
            loss_weight_override (float, optional):
                The weight of loss used to
                override the original weight of loss.
                Defaults to None.
            reduction_override (Literal['mean', 'sum', 'none'], optional)::
                The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override \
            if reduction_override is not None \
            else self.reduction
        loss_weight = loss_weight_override\
            if loss_weight_override is not None \
            else self.loss_weight

        if self.use_merged:
            pose_prior_loss = self.merged_log_likelihood(body_pose)
        else:
            pose_prior_loss = self.log_likelihood(body_pose)

        pose_prior_loss = loss_weight * pose_prior_loss

        if reduction == 'mean':
            pose_prior_loss = pose_prior_loss.mean()
        elif reduction == 'sum':
            pose_prior_loss = pose_prior_loss.sum()
        return pose_prior_loss
