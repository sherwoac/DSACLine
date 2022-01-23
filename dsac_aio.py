import torch
import torch.nn.functional as F
import random

import dsac


class DsacAio(dsac.DSAC):
    """
    Differentiable RANSAC to robustly fit lines. Attempt to leverage vector ops
    based on: https://github.com/vislearn/DSACLine
    """
    def __init__(self, hyps, inlier_thresh, inlier_beta, inlier_alpha, loss_function, random_generator=None):
        super().__init__(hyps, inlier_thresh, inlier_beta, inlier_alpha, loss_function, random_generator)

    def calculate_loss(self, prediction, labels):
        """
        Perform robust, differentiable line fitting according to DSAC.

        Returns the expected loss of choosing a good line hypothesis which can be used for backprob.

        prediction -- predicted 2D points for a batch of images, array of shape (Bx2) where
            B is the number of images in the batch
            2 is the number of point dimensions (y, x)
        labels -- ground truth labels for the batch, array of shape (Bx2) where
            B is the number of images in the batch
            2 is the number of parameters (intercept, slope)
        """
        batch_size = prediction.size(0)
        number_of_inliers = prediction.size(2)

        y_b_p = prediction[:, 0]  # all y-values of the prediction
        x_b_p = prediction[:, 1]  # all x.values of the prediction

        slopes_b_h, intercepts_b_h = self._sample_hyp(x_b_p, y_b_p)
        batch_hypothesis_scores, inliers_b_h_p = self._soft_inlier_count(slopes_b_h, intercepts_b_h, x_b_p, y_b_p)
        slopes_b_h, intercepts_b_h = self._refine_hyp(x_b_p, y_b_p, inliers_b_h_p)

        model_hypothesis = torch.stack([slopes_b_h, intercepts_b_h], dim=-1)
        losses_b_h = self.loss_function.get_loss(model_hypothesis, labels.squeeze())
        hyp_scores = F.softmax(self.inlier_alpha * batch_hypothesis_scores, 0)
        exp_loss = torch.sum(losses_b_h * hyp_scores)

        # assemble losses based on top scores
        top_loss_locations_b = torch.argmax(hyp_scores, dim=-1, keepdim=True)
        top_loss = torch.gather(losses_b_h,
                                dim=1,
                                index=top_loss_locations_b).sum()
        """
        store top:
           - loss
           - model params: slope and intercept
           - inliers
        """
        self.est_losses = top_loss.detach()
        # nice little lesson in gather indexing - need to repeat index to get both values (b x h x 2) out of last dimension
        top_loss_locations_for_both_model_parameters = top_loss_locations_b.unsqueeze(-1).repeat((1, 1, 2))
        self.est_parameters = torch.gather(model_hypothesis,
                                           dim=1,
                                           index=top_loss_locations_for_both_model_parameters).squeeze().detach()
        top_loss_locations_for_all_inliers = top_loss_locations_b.unsqueeze(-1).repeat((1, 1, number_of_inliers))
        self.batch_inliers = torch.gather(inliers_b_h_p,
                                          dim=1,
                                          index=top_loss_locations_for_all_inliers).squeeze().detach()

        return exp_loss / batch_size, top_loss / batch_size

    def _sample_hyp(self,
                    x: torch.FloatTensor,  # b x p
                    y: torch.FloatTensor,  # b x p
                    x_threshold: float = 0.01) \
            -> (torch.FloatTensor, torch.FloatTensor):
        """
        Calculate number_of_required_hypothesis hypotheses (slope, intercept) from two random points (x, y).

        x -- vector of x values
        y -- vector of y values
        """
        num_batches = x.size(0)
        num_correspondences = x.size(-1)

        xs = (x.unsqueeze(-2) - x.unsqueeze(-1)).reshape((num_batches, -1))  # b x (p**2) all combinations of x
        ys = (y.unsqueeze(-2) - y.unsqueeze(-1)).reshape((num_batches, -1))  # b x (p**2)
        slopes = ys.divide(xs)  # b x (p**2)
        intercepts = y.repeat((1, num_correspondences)) - slopes * x.repeat((1, num_correspondences))  # b x (p**2)
        acceptance_criteria = ~(torch.abs(xs) < x_threshold) * ~slopes.isnan()  # b x (p**2)

        chosen_accepted_indices = torch.multinomial(acceptance_criteria.double(),  # b x h
                                                    self.hyps,  # number_of_required_hypotheses
                                                    replacement=False,  # no repeated hypothesis
                                                    generator=self.random_generator)

        chosen_slopes = torch.gather(slopes,  # b x h
                                     dim=1,
                                     index=chosen_accepted_indices)

        chosen_intercepts = torch.gather(intercepts,  # b x h
                                         dim=1,
                                         index=chosen_accepted_indices)

        return chosen_slopes, chosen_intercepts

    def _soft_inlier_count(self,
                           slopes: torch.FloatTensor,  # b x h
                           intercepts: torch.FloatTensor,  # b x h
                           xs: torch.FloatTensor,  # b x p
                           ys: torch.FloatTensor) \
            -> (torch.Tensor, torch.Tensor):  # b x p
        """
        distance d of point p to a line given by:
        y = ax + b
        is given by:
        d = \frac{a \cross \left(b - p \right){ a \dot a}

        :return:
        """

        # point line distances
        dists_b_h_p = torch.abs(slopes.unsqueeze(dim=-1) * xs.unsqueeze(-2) - ys.unsqueeze(-2) + intercepts.unsqueeze(dim=-1))
        dists_b_h_p = torch.divide(dists_b_h_p, torch.sqrt(slopes * slopes + 1).unsqueeze(-1))

        # soft inliers
        dists_b_h_p = 1 - torch.sigmoid(self.inlier_beta * (dists_b_h_p - self.inlier_thresh))
        scores_b_h = torch.sum(dists_b_h_p, dim=[-1])

        return scores_b_h, dists_b_h_p

    @staticmethod
    def _refine_hyp(x_b_p, y_b_p, weights_b_h_p):
        """
        Refinement by weighted Deming regression.

        Fits a line minimizing errors in x and y, implementation according to:
            'Performance of Deming regression analysis in case of misspecified
            analytical error ratio in method comparison studies'
            Kristian Linnet, in Clinical Chemistry, 1998

        x -- vector of x values, b x p
        y -- vector of y values, b x p
        weights -- vector of weights (1 per point), b x h x p
        return: slopes, intercepts b x h
        """
        ws = weights_b_h_p.sum(dim=-1)
        xm_b_h = (x_b_p.unsqueeze(-2) * weights_b_h_p).sum(dim=-1) / ws
        ym_b_h = (y_b_p.unsqueeze(-2) * weights_b_h_p).sum(dim=-1) / ws

        u = torch.pow(x_b_p.unsqueeze(dim=-2) - xm_b_h.unsqueeze(dim=-1), 2)
        u = (u * weights_b_h_p).sum(dim=-1)

        q = torch.pow(y_b_p.unsqueeze(dim=-2) - ym_b_h.unsqueeze(dim=-1), 2)
        q = (q * weights_b_h_p).sum(dim=-1)

        p = torch.mul(x_b_p.unsqueeze(dim=-2) - xm_b_h.unsqueeze(dim=-1), y_b_p.unsqueeze(dim=-2) - ym_b_h.unsqueeze(dim=-1))
        p = (p * weights_b_h_p).sum(dim=-1)

        slopes = (q - u + torch.sqrt(torch.pow(u - q, 2) + 4 * torch.pow(p, 2))) / (2 * p)
        intercepts = ym_b_h - slopes * xm_b_h

        return slopes, intercepts
