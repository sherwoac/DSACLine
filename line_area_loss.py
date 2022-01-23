import torch
import line_loss


class LineLossArea(line_loss.LineLoss):
    """
    Compares two lines by calculating the area between them, squared
    """

    def __init__(self, image_size: int):
        """
        Constructor.

        image_size -- size of the input images, used to normalize the loss
        """
        super().__init__(image_size)

    def get_loss(self, est: torch.FloatTensor, gt: torch.FloatTensor) -> torch.FloatTensor:
        """
        calculate the square of the area between the lines
        :param est: b x h x [intercept, slope]
        :param gt: b x [intercept, slope]
        :return: sum of the squared areas between the lines, min-ed with 1, the max squared area?
        """
        def get_squared_area(a_1: torch.FloatTensor,
                             b_1: torch.FloatTensor,
                             a_2: torch.FloatTensor,
                             b_2: torch.FloatTensor) -> torch.FloatTensor:
            """
            calculate area between lines, squared, from Mathematica:
                \!\(\*SubsuperscriptBox[\(a\), \(1\), \(2\)]\)/3 - (
                    2 Subscript[a, 1] Subscript[a, 2])/3 +
                \!\(\*SubsuperscriptBox[\(a\), \(2\), \(2\)]\)/3 +
                Subscript[a, 1] Subscript[b, 1] - Subscript[a, 2] Subscript[b, 1] +
                \!\(\*SubsuperscriptBox[\(b\), \(1\), \(2\)]\) -
                Subscript[a, 1] Subscript[b, 2] + Subscript[a, 2] Subscript[b, 2] -
                2 Subscript[b, 1] Subscript[b, 2] +
                \!\(\*SubsuperscriptBox[\(b\), \(2\), \(2\)]\)
            """
            return (a_1 * a_1 - 2. * a_1 * a_2 + a_2 * a_2) / 3. + \
                   a_1 * b_1 - a_2 * b_1 + b_1 * b_1 - a_1 * b_2 + a_2 * b_2 - 2 * b_1 * b_2 + b_2 * b_2

        area_squared_b_h = get_squared_area(est[:, :, 1],
                                        est[:, :, 0],
                                        gt[:, 1].unsqueeze(-1),
                                        gt[:, 0].unsqueeze(-1))

        return torch.minimum(area_squared_b_h, torch.tensor([1.]).to(area_squared_b_h.device))

