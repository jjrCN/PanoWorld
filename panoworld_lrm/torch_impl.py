import torch
import torch.nn.functional as F
from torch import Tensor

def _eval_sh_bases_fast(basis_dim: int, dirs: Tensor):
    """
    Evaluate spherical harmonics bases at unit direction for high orders
    using approach described by
    Efficient Spherical Harmonic Evaluation, Peter-Pike Sloan, JCGT 2013
    https://jcgt.org/published/0002/02/06/


    :param basis_dim: int SH basis dim. Currently, only 1-25 square numbers supported
    :param dirs: torch.Tensor (..., 3) unit directions

    :return: torch.Tensor (..., basis_dim)

    See reference C++ code in https://jcgt.org/published/0002/02/06/code.zip
    """
    result = torch.empty(
        (*dirs.shape[:-1], basis_dim), dtype=dirs.dtype, device=dirs.device
    )

    result[..., 0] = 0.2820947917738781

    if basis_dim <= 1:
        return result

    x, y, z = dirs.unbind(-1)

    fTmpA = -0.48860251190292
    result[..., 2] = -fTmpA * z
    result[..., 3] = fTmpA * x
    result[..., 1] = fTmpA * y
    if basis_dim <= 4:
        return result

    z2 = z * z
    fTmpB = -1.092548430592079 * z
    fTmpA = 0.5462742152960395
    fC1 = x * x - y * y
    fS1 = 2 * x * y
    result[..., 6] = 0.9461746957575601 * z2 - 0.3153915652525201
    result[..., 7] = fTmpB * x
    result[..., 5] = fTmpB * y
    result[..., 8] = fTmpA * fC1
    result[..., 4] = fTmpA * fS1

    if basis_dim <= 9:
        return result

    fTmpC = -2.285228997322329 * z2 + 0.4570457994644658
    fTmpB = 1.445305721320277 * z
    fTmpA = -0.5900435899266435
    fC2 = x * fC1 - y * fS1
    fS2 = x * fS1 + y * fC1
    result[..., 12] = z * (1.865881662950577 * z2 - 1.119528997770346)
    result[..., 13] = fTmpC * x
    result[..., 11] = fTmpC * y
    result[..., 14] = fTmpB * fC1
    result[..., 10] = fTmpB * fS1
    result[..., 15] = fTmpA * fC2
    result[..., 9] = fTmpA * fS2

    if basis_dim <= 16:
        return result

    fTmpD = z * (-4.683325804901025 * z2 + 2.007139630671868)
    fTmpC = 3.31161143515146 * z2 - 0.47308734787878
    fTmpB = -1.770130769779931 * z
    fTmpA = 0.6258357354491763
    fC3 = x * fC2 - y * fS2
    fS3 = x * fS2 + y * fC2
    result[..., 20] = 1.984313483298443 * z2 * (
        1.865881662950577 * z2 - 1.119528997770346
    ) + -1.006230589874905 * (0.9461746957575601 * z2 - 0.3153915652525201)
    result[..., 21] = fTmpD * x
    result[..., 19] = fTmpD * y
    result[..., 22] = fTmpC * fC1
    result[..., 18] = fTmpC * fS1
    result[..., 23] = fTmpB * fC2
    result[..., 17] = fTmpB * fS2
    result[..., 24] = fTmpA * fC3
    result[..., 16] = fTmpA * fS3
    return result


def _spherical_harmonics(
    degrees_to_use: int,
    dirs: torch.Tensor,  # [..., 3]
    coeffs: torch.Tensor,  # [..., K, 3] or [..., K, 1]
):
    """Pytorch implementation of `gsplat.cuda._wrapper.spherical_harmonics()`."""
    assert (degrees_to_use + 1) ** 2 <= coeffs.shape[-2], coeffs.shape
    batch_dims = dirs.shape[:-1]
    assert dirs.shape == batch_dims + (3,), dirs.shape
    assert (
        (len(coeffs.shape) == len(batch_dims) + 2)
        and coeffs.shape[:-2] == batch_dims
        and (coeffs.shape[-1] == 3 or coeffs.shape[-1] == 1)
    ), coeffs.shape
    dirs = F.normalize(dirs, p=2, dim=-1)
    num_bases = (degrees_to_use + 1) ** 2
    bases = torch.zeros_like(coeffs[..., 0])
    bases[..., :num_bases] = _eval_sh_bases_fast(num_bases, dirs)
    return (bases[..., None] * coeffs).sum(dim=-2)
