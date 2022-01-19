import torch
import torch.nn.functional as F
import random

import dsac


class DsacAio(dsac.DSAC):
    """
    Differentiable RANSAC to robustly fit lines. Attempt to leverage vector ops
    based on: https://github.com/vislearn/DSACLine
    """
    def __init__(self, hyps, inlier_thresh, inlier_beta, inlier_alpha, loss_function, random_generator):
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

        # avg_exp_loss = 0 # expected loss
        avg_exp_loss = torch.tensor(0., requires_grad=True)
        # avg_top_loss = 0 # loss of best hypothesis
        avg_top_loss = torch.tensor(0., requires_grad=True)

        self.est_parameters = torch.zeros(batch_size, 2) # estimated lines
        self.est_losses = torch.zeros(batch_size) # loss of estimated lines

        self.batch_inliers = torch.zeros(batch_size, number_of_inliers) # (soft) inliers for estimated lines

        hyp_losses = torch.zeros([self.hyps, 1], requires_grad=True) # loss of each hypothesis
        hyp_scores = torch.zeros([self.hyps, 1], requires_grad=True) # score of each hypothesis

        y = prediction[:, 0]  # all y-values of the prediction
        x = prediction[:, 1]  # all x.values of the prediction

        slopes, intercepts = self._sample_hyp(x, y)
        score, inliers = self._soft_inlier_count(slopes, intercepts, x, y)

        for b in range(0, batch_size):

            max_score = 0 	# score of best hypothesis

            for h in range(0, self.hyps):
                # === step 3: refine hypothesis ===========================
                slope, intercept = self._refine_hyp(x, y, inliers)

                hyp = torch.zeros([2])
                hyp[1] = 0. #slope
                hyp[0] = 0. #intercept

                # === step 4: calculate loss of hypothesis ================
                loss = self.loss_function(hyp, labels[b])

                # store results
                hyp_losses.data[h] = loss
                hyp_scores.data[h] = score

                # keep track of best hypothesis so far
                if score > max_score:
                    max_score = score
                    self.est_losses[b] = loss
                    self.est_parameters[b] = hyp
                    self.batch_inliers[b] = inliers

            # === step 5: calculate the expectation ===========================

            #softmax distribution from hypotheses scores
            hyp_scores = F.softmax(self.inlier_alpha * hyp_scores, 0)

            # expectation of loss
            exp_loss = torch.sum(hyp_losses * hyp_scores)
            print(exp_loss.grad)
            avg_exp_loss = avg_exp_loss + exp_loss

            # loss of best hypothesis (for evaluation)
            avg_top_loss = avg_top_loss + self.est_losses[b]
        print(avg_exp_loss.grad)
        return avg_exp_loss / batch_size, avg_top_loss / batch_size

    def _sample_hyp(self,
                    x: torch.FloatTensor,  # b x p
                    y: torch.FloatTensor,  # b x p
                    x_threshold: float = 0.01) \
            -> (torch.Tensor, torch.Tensor):
        """
        Calculate number_of_required_hypothesis hypotheses (slope, intercept) from two random points (x, y).

        x -- vector of x values
        y -- vector of y values
        """
        num_correspondences = x.size(-1)

        xs = (x.unsqueeze(-2) - x.unsqueeze(-1)).reshape((2, -1))  # b x (p**2) all combinations of x
        ys = (y.unsqueeze(-2) - y.unsqueeze(-1)).reshape((2, -1))  # b x (p**2)
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
        dists = torch.abs(slopes.unsqueeze(dim=-1) * xs - ys + intercepts.unsqueeze(dim=-1))
        dists = torch.divide(dists, torch.sqrt(slopes * slopes + 1).unsqueeze(-1))

        # soft inliers
        dists = 1 - torch.sigmoid(self.inlier_beta * (dists - self.inlier_thresh))
        scores = torch.sum(dists, dim=[-1])

        return scores, dists
