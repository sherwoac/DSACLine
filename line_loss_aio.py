import torch
import line_loss


class LineLossAio(line_loss.LineLoss):
    '''
    Compares two lines by calculating the distance between their ends in the image.
    '''

    def __init__(self, image_size: int):
        '''
        Constructor.

        image_size -- size of the input images, used to normalize the loss
        '''
        super().__init__(image_size)

    @staticmethod
    def _get_unit_square_intercepts(slopes: torch.Tensor, intercepts: torch.Tensor) -> torch.Tensor:
        """
        returns unit square intercepts for given slope (a) and intercepts (b)
        y = ax + b
        solves:
        left: y = b
        x = 0
        y = intercept
        right: y = a + b
        x = 1
        y = slopes + intercept
        bottom: 0 = ax + b
        x = torch.divide(- intercept, slopes)
        y = 0
        top: 1 = ax + b
        x = torch.divide(1 - intercept, slopes)
        y = 1

        :param slopes: b x 1
        :param intercepts: b x 1
        :return: points where line intersects unit square borders: b x 2 pts of [[x_1, y_1], [x_2, y_2]]: b x 2 x 2
        """
        assert len(slopes.size()) == 1, "_get_unit_square_intercepts doesn't work with more than two dimensions"
        batches = slopes.size(0)
        x = torch.column_stack([torch.zeros(batches).to(slopes.device),     # x = 0
                                torch.ones(batches).to(slopes.device),      # x = 1
                                torch.divide(-1. * intercepts, slopes + .00001),     # y = 0
                                torch.divide(1. - intercepts, slopes + .00001)])     # y = 1

        y = torch.column_stack([intercepts,                                 # x = 0
                                slopes + intercepts,                        # x = 1
                                torch.zeros(batches).to(slopes.device),     # y = 0
                                torch.ones(batches).to(slopes.device)])     # y = 1

        hits_unit_square = (y >= 0.) * (y <= 1.) * (x >= 0.) * (x <= 1.)

        # this bit was added to allow for missing the unit square, to replicate original functionality
        good_lines = torch.sum(hits_unit_square, dim=-1, keepdim=True) == 2
        big_slopes = torch.abs(slopes.unsqueeze(-1)) > 1
        x_lines = torch.column_stack([torch.ones(batches, 2).bool().to(slopes.device),
                                      torch.zeros(batches, 2).bool().to(slopes.device)])

        y_lines = torch.column_stack([torch.zeros(batches, 2).bool().to(slopes.device),
                                      torch.ones(batches, 2).bool().to(slopes.device)])

        acceptance = \
            hits_unit_square * good_lines + \
            big_slopes * ~good_lines * y_lines + \
            ~big_slopes * ~good_lines * x_lines

        # debug this line with: torch.where(~(torch.sum(acceptance, dim=-1) == 2))
        assert \
            torch.sum(torch.sum(acceptance, dim=-1, keepdim=True) == 2) == batches, \
            f"should be equal: acceptance: {acceptance} batches: {batches}"

        return torch.column_stack((x[acceptance], y[acceptance])).reshape(batches, 2, 2)

    def get_loss(self, est: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """
        Calculate the line loss:
        currently finds the distance between the left most points, the right most points and adds

        est -- estimated line, form: b x 2: b x [intercept, slope]
        gt -- ground truth line, form: b x 2: [intercept, slope]
        """
        batches = gt.size(0)
        if len(est.size()) == 3:
            # flatten to b * h then reinflate
            v_est = est.view(est.size(0) * est.size(1), -1)
        else:
            v_est = est.view()

        pts_est = self._get_unit_square_intercepts(v_est[:, 1], v_est[:, 0])
        pts_gt = self._get_unit_square_intercepts(gt[:, 1], gt[:, 0])
        pts_est = pts_est.reshape(batches, -1, 2, 2)
        v_pts_gt = pts_gt.view(batches, 1, 2, 2)
        same_points_compared = torch.linalg.vector_norm(pts_est - v_pts_gt, dim=-1, ord=2).sum(-1)
        pts_est_swapped = pts_est.index_select(-2, torch.LongTensor([1, 0]).to(est.device))
        opposite_points_compared = torch.linalg.vector_norm(pts_est_swapped - v_pts_gt, dim=-1, ord=2).sum(-1)
        mins = torch.minimum(same_points_compared, opposite_points_compared) * self.image_size
        assert not mins.isnan().any()
        return mins


