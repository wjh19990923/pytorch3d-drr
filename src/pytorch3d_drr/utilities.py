import torch


def determine_device(device=None):
    if device is None:
        if torch.cuda.is_available():
            cuda_idx = torch.cuda.current_device()
            device = torch.device("cuda", cuda_idx)
        else:
            device = torch.device("cpu")
    else:
        if not torch.cuda.is_available():
            device = torch.device("cpu")
        else:
            device = torch.device(device)

    return device


def kneefit_to_pytorch3d_renderer(intrinsic_affine_mat: torch.Tensor, tmat: torch.Tensor, calDict: dict, duplaOrigin=False):
    """because of the differences between kneefit and our rendering setup, we need to do the
    Args:
        intrinsic_affine_mat: normally anatomy.original_intrinsic_affine_mat()
        shape is Bx4x4
    following adjustments
        - adjust the origin from middle of the volume to corner of the volume
        - the axes were defined differently
        - somehow we need an additional world transform when using with a camera set up according to
          the calibration data
    Returns:
        new mat and additionalWorldTransform that you can pass to
        femurCT.set_model_matrix(newTMat, is_yours=False, theirs_to_yours=additionalWorldT)
        .. note:: original data is not modified

    For an example, look at `test_renderer.py::test_kneefit`
    """
    assert intrinsic_affine_mat.size()[0] == tmat.size()[0]
    dev = tmat.device
    batchsize = tmat.size()[0]
    # axes are defined differently
    signXY = -1.
    if duplaOrigin:
        signXY = 1.
    coordinateSystemT = torch.tensor(
        [[signXY, 0, 0, 0],
         [0, signXY, 0, 0],
         [0, 0, 1, 0],
         [0, 0, 0, 1]]).expand(batchsize, 4, 4).to(dev)
    worldT = torch.zeros(batchsize, 4, 4).to(dev)
    worldT[:, 0, 0] = 1.0
    worldT[:, 1, 1] = 1.0
    worldT[:, 2, 2] = 1.0
    worldT[:, 3, 3] = 1.0
    worldT[:, 0:3, 3] = torch.stack((
        calDict["cal_principalp_x"],
        calDict["cal_principalp_y"],
        calDict["cal_focal_length"]), axis=1)
    # kneefit puts the origin in the center of the volume, while pytorch3d expects it to be at the
    # corner. This mismatch needs to be corrected on the original volume, before grid reduction
    # so we need the original_intrinsic_affine_mat
    centerTransform = torch.zeros(batchsize, 4, 4).to(dev)
    centerTransform[:, 0, 0] = 1.0
    centerTransform[:, 1, 1] = 1.0
    centerTransform[:, 2, 2] = 1.0
    centerTransform[:, 3, 3] = 1.0
    if not duplaOrigin:
        centerTransform[:, 0:3, 3] = (
            torch.tensor([[1, 1, -1]]).to(dev) * intrinsic_affine_mat[:, 0:3, 3]
        )
    newTmat = tmat @ centerTransform @ coordinateSystemT
    return newTmat, worldT
