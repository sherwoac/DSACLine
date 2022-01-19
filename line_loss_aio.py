import torch
import line_loss

class LineLossAio(line_loss.LineLoss):
    '''
    Compares two lines by calculating the distance between their ends in the image.
    '''

    def __init__(self, image_size):
        '''
        Constructor.

        image_size -- size of the input images, used to normalize the loss
        '''
        super().__init__(image_size)

    def _get_unit_square_intercepts(self, slopes, intercept):
        """
        returns unit square intercepts for given slope (a) and intercepts (b)
        y = ax + b
        solves:
        right: y = a + b
        x = 1
        y = slopes + intercept
        left: y = b
        x = 0
        y = intercept
        top: 1 = ax + b
        x = torch.divide(1 - intercept, slopes)
        y = 1
        bottom: 0 = ax + b
        x = torch.divide(- intercept, slopes)
        y = 0

        :param slopes: b x 1
        :param intercepts: b x 1
        :return: points where line intersects unit square borders: b x 2 pts of [x, y]: b x 2 x 2
        """
        batches = slopes.size(0)
        x = torch.column_stack([torch.ones(batches),
                                torch.zeros(batches),
                                torch.divide(1 - intercept, slopes),
                                torch.divide(-1 * intercept, slopes)])

        y = torch.column_stack([slopes + intercept,
                                intercept,
                                torch.ones(batches),
                                torch.zeros(batches)])

        acceptance = (y >= 0) * (y <= 1) * (x >= 0) * (x <= 1)
        return torch.column_stack((x[acceptance].reshape(batches, 1, -1), y[acceptance].reshape(batches, 1, -1)))  # b x pts(x, y)

    def get_line_loss(self, est, gt):
        '''
        Calculate the line loss.

        est -- estimated line, form: b x 2: b x [intercept, slope]
        gt -- ground truth line, form: b x 2: [intercept, slope]
        '''

        pts_est = self._get_unit_square_intercepts(est[:, 1], est[:, 0])
        pts_gt = self._get_unit_square_intercepts(gt[:, 1], gt[:, 0])

        # not clear which ends of the lines should be compared (there are ambigious cases), compute both and take min
        loss1 = pts_est - pts_gt
        loss1 = loss1.norm(2, 1).sum()

        # flip_mat = torch.zeros([2, 2])
        # flip_mat.data[0, 1] = 1
        # flip_mat[1, 0] = 1
        flip_mat = torch.eye(2).flip(dims=[1]).unsqueeze(0)
        loss2 = pts_est - torch.matmul(flip_mat, pts_gt)
        loss2 = loss2.norm(2, 1).sum()

        return torch.min(torch.Tensor([loss1, loss2])) * self.image_size
