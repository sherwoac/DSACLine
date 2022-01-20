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
    def _get_unit_square_intercepts(slopes: torch.FloatTensor, intercepts: torch.FloatTensor) -> torch.FloatTensor:
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
        :return: leftmost ordered points where line intersects unit square borders: b x 2 pts of [x, y]: b x 2 x 2
        """
        batches = slopes.size(0)
        x = torch.column_stack([torch.zeros(batches),
                                torch.ones(batches),
                                torch.divide(-1. * intercepts, slopes),
                                torch.divide(1. - intercepts, slopes)])

        y = torch.column_stack([intercepts,
                                slopes + intercepts,
                                torch.zeros(batches),
                                torch.ones(batches)])

        acceptance = (y >= 0.) * (y <= 1.) * (x >= 0.) * (x <= 1.)
        leftmost = torch.argmin(x[acceptance], keepdim=True)
        rightmost = torch.argmax(x[acceptance], keepdim=True)
        xs = torch.column_stack((x[acceptance][leftmost], x[acceptance][rightmost]))
        ys = torch.column_stack((y[acceptance][leftmost], y[acceptance][rightmost]))
        return torch.row_stack((xs, ys)).reshape(batches, 2, 2)

    def get_line_loss(self, est: torch.FloatTensor, gt: torch.FloatTensor) -> torch.FloatTensor:
        """
        Calculate the line loss:
        currently finds the distance between the left most points, the right most points and adds

        est -- estimated line, form: b x 2: b x [intercept, slope]
        gt -- ground truth line, form: b x 2: [intercept, slope]
        """
        pts_est = self._get_unit_square_intercepts(est[:, 1], est[:, 0])
        pts_gt = self._get_unit_square_intercepts(gt[:, 1], gt[:, 0])

        return (pts_est - pts_gt).norm(2, 1).sum() * self.image_size
