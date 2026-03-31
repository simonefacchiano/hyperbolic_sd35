# ---------------------------------------
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# ---------------------------------------

# Modified from github.com/facebookresearch/meru

"""
Implementation of common operations for the Lorentz model of hyperbolic geometry.
This model represents a hyperbolic space of `d` dimensions on the upper-half of
a two-sheeted hyperboloid in a Euclidean space of `(d+1)` dimensions.

Hyperbolic geometry has a direct connection to the study of special relativity
theory -- implementations in this module borrow some of its terminology. The axis
of symmetry of the Hyperboloid is called the _time dimension_, while all other
axes are collectively called _space dimensions_.

All functions implemented here only input/output the space components, while
while calculating the time component according to the Hyperboloid constraint:

    `x_time = torch.sqrt(1 / curv + torch.norm(x_space) ** 2)`
"""
from __future__ import annotations

import math

import torch
from torch import Tensor

from typing import Union, Tuple
from loguru import logger


def get_root_features(
    curvature: float = 1.0,
    feature_dim: int = 512,
    device: Union[str, torch.device] = "cpu",
) -> Tensor:
    """
    Get the root point on the hyperboloid.

    Args:
        curvature: Positive scalar denoting negative hyperboloid curvature.
        feature_dim: Dimensionality of the space components of the hyperboloid points.
        device: Device to create the tensor on.

    Returns:
        Tensor of shape `(feature_dim)` giving the space components of
        the root point on the hyperboloid.
    """

    root_space = torch.zeros((feature_dim), device=device)
    return root_space


def to_poincare(x: Tensor, curv: float = 1.0, eps: float = 1e-8) -> Tensor:
    """
    Map points from the Lorentz model to the Poincare ball model.

    Args:
        x: Tensor of shape `(B, D)` giving a batch of space components of
            vectors on the hyperboloid.
        curv: Positive scalar denoting negative hyperboloid curvature.
        eps: Small float number to avoid numerical instability.
    Returns:
        Tensor of shape `(B, D)` giving the mapped points in the Poincare ball.
    """

    x_time = compute_time_component(x, curv)
    curv_radius = 1 / math.sqrt(curv)
    denom = x_time + curv_radius
    poincare_x = x / denom
    return poincare_x


def from_poincare(x: Tensor, curv: float = 1.0, eps: float = 1e-8) -> Tensor:
    """
    Map points from the Poincare ball model to the Lorentz model.

    Args:
        x: Tensor of shape `(B, D)` giving a batch of points in the Poincare ball.
        curv: Positive scalar denoting negative hyperboloid curvature.
        eps: Small float number to avoid numerical instability.
    Returns:
        Tensor of shape `(B, D)` giving the space components of the corresponding
        points on the hyperboloid.
    """
    # Compute squared norm
    x_norm_squared = torch.sum(x * x, dim=-1, keepdim=True)

    # Handle numerical stability
    x_norm_squared = torch.clamp(x_norm_squared, min=0.0, max=1.0 / curv - eps)

    # Compute the denominator
    factor = 1.0 / (1.0 - curv * x_norm_squared + eps)

    # Scale the Poincare points to get the space components of Lorentz vectors
    lorentz_space = 2 * x * factor

    return lorentz_space


def compute_time_component(x: Tensor, curv: float | Tensor = 1.0) -> Tensor:
    """
    Given the space components of points on the hyperboloid, compute the time
    component according to the hyperboloid constraint.

    Args:
        x: Tensor of shape `(B, D)` giving a batch of space components of
            vectors on the hyperboloid.
        curv: Positive scalar denoting negative hyperboloid curvature.

    Returns:
        Tensor of shape `(B, 1)` giving the time component of input vectors.
    """

    x_time = torch.sqrt(1 / curv + torch.sum(x**2, dim=-1, keepdim=True))
    return x_time


def verify_points_on_hyperboloid(
    points: Tensor, curv: float = 1.0, eps: float = 1e-5
) -> Tensor:
    """
    Verify if points are on the hyperboloid with given curvature.

    Args:
        points: Tensor of shape (N, D) giving space components of points
        curv: Curvature parameter (default 1.0)
        eps: Tolerance for numerical precision

    Returns:
        Boolean tensor of shape (N,) indicating which points satisfy the constraint
    """
    # Calculate the time components
    space_norm_squared = torch.sum(points**2, dim=-1)
    time_component = torch.sqrt(1 / curv + space_norm_squared)

    # Calculate Minkowski inner product with itself (should be 1/curv for points on hyperboloid)
    minkowski_norm = time_component**2 - space_norm_squared

    # Check if points satisfy the hyperboloid constraint within tolerance
    return torch.abs(minkowski_norm - 1 / curv) < eps


def pairwise_inner(x: Tensor, y: Tensor, curv: float | Tensor = 1.0):
    """
    Compute pairwise Lorentzian inner product between input vectors.

    Args:
        x: Tensor of shape `(B1, D)` giving a space components of a batch
            of vectors on the hyperboloid.
        y: Tensor of shape `(B2, D)` giving a space components of another
            batch of points on the hyperboloid.
        curv: Positive scalar denoting negative hyperboloid curvature.
        eps: Small float number to avoid numerical instability.

    Returns:
        Tensor of shape `(B1, B2)` giving pairwise Lorentzian inner product
        between input vectors.
    """

    x_time = torch.sqrt(1 / curv + torch.sum(x**2, dim=-1, keepdim=True)).to(x.device)
    y_time = torch.sqrt(1 / curv + torch.sum(y**2, dim=-1, keepdim=True)).to(y.device)
    xyl = x @ y.T - x_time @ y_time.T
    return xyl


def pairwise_dist(
    x: Tensor, y: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-8
) -> Tensor:
    """
    Compute the pairwise geodesic distance between two batches of points on
    the hyperboloid.

    Args:
        x: Tensor of shape `(B1, D)` giving a space components of a batch
            of point on the hyperboloid.
        y: Tensor of shape `(B2, D)` giving a space components of another
            batch of points on the hyperboloid.
        curv: Positive scalar denoting negative hyperboloid curvature.
        eps: Small float number to avoid numerical instability.

    Returns:
        Tensor of shape `(B1, B2)` giving pairwise distance along the geodesics
        connecting the input points.
    """

    # Ensure numerical stability in arc-cosh by clamping input.
    c_xyl = -curv * pairwise_inner(x, y, curv)
    _distance = torch.acosh(torch.clamp(c_xyl, min=1 + eps))
    return _distance / curv**0.5


def exp_map0(x: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-8) -> Tensor:
    """
    Map points from the tangent space at the vertex of hyperboloid, on to the
    hyperboloid. This mapping is done using the exponential map of Lorentz model.

    Args:
        x: Tensor of shape `(B, D)` giving batch of Euclidean vectors to project
            onto the hyperboloid. These vectors are interpreted as velocity
            vectors in the tangent space at the hyperboloid vertex.
        curv: Positive scalar denoting negative hyperboloid curvature.
        eps: Small float number to avoid division by zero.

    Returns:
        Tensor of same shape as `x`, giving space components of the mapped
        vectors on the hyperboloid.
    """

    rc_xnorm = curv**0.5 * torch.norm(x, dim=-1, keepdim=True)

    # Ensure numerical stability in sinh by clamping input.
    sinh_input = torch.clamp(rc_xnorm, min=eps, max=math.asinh(2**15))
    _output = torch.sinh(sinh_input) * x / torch.clamp(rc_xnorm, min=eps)
    return _output


def log_map0(x: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-8) -> Tensor:
    """
    Inverse of the exponential map: map points from the hyperboloid on to the
    tangent space at the vertex, using the logarithmic map of Lorentz model.

    Args:
        x: Tensor of shape `(B, D)` giving space components of points
            on the hyperboloid.
        curv: Positive scalar denoting negative hyperboloid curvature.
        eps: Small float number to avoid division by zero.

    Returns:
        Tensor of same shape as `x`, giving Euclidean vectors in the tangent
        space of the hyperboloid vertex.
    """

    # Calculate distance of vectors to the hyperboloid vertex.
    rc_x_time = torch.sqrt(1 + curv * torch.sum(x**2, dim=-1, keepdim=True))
    _distance0 = torch.acosh(torch.clamp(rc_x_time, min=1 + eps))

    rc_xnorm = curv**0.5 * torch.norm(x, dim=-1, keepdim=True)
    _output = _distance0 * x / torch.clamp(rc_xnorm, min=eps)
    return _output


def log_map(
    x: Tensor, y: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-8
) -> Tensor:
    """
    Hyperbolic Displacement Vector $\delta = \operatorname{log}_{x}^\kappa(y)$.
    Maps a point y from the hyperboloid to the tangent space at point x.

    Args:
        x: Tensor of shape `(B, D)` giving space components of the base point.
        y: Tensor of shape `(B, D)` giving space components of the target point.
        curv: Positive scalar or tensor denoting negative hyperboloid curvature.
        eps: Small float number to avoid numerical instability.

    Returns:
        Tensor representing the tangent vector at x that points toward y.

    Note:
        Assumes inputs are aligned for element-wise operation.
    """
    # Get the distance between x and y, properly scaled by curvature
    dist = pairwise_dist(x, y, curv, eps)

    # Get the Lorentzian inner product
    inner = pairwise_inner(x, y, curv)

    # Early return for points that are very close
    if torch.all(dist < eps):
        return torch.zeros_like(x)

    # Project y onto the tangent space at x
    # The projection is: y - <x,y>_L * x
    v = y - inner * x

    # Early return if the projection is near zero
    v_norm = torch.norm(v, dim=-1, keepdim=True)
    if torch.all(v_norm < eps):
        return torch.zeros_like(x)

    # Calculate scaling factor
    # Note: pairwise_dist already returns dist / sqrt(curv)
    sqrt_c = torch.sqrt(curv)

    # We need sinh(sqrt(c) * dist)
    # Since dist = acosh(-c<x,y>_L) / sqrt(c)
    # sinh(sqrt(c) * dist) = sinh(acosh(-c<x,y>_L)) = sqrt((-c<x,y>_L)^2 - 1)
    sinh_term = torch.sinh(sqrt_c * dist)

    # Avoid division by zero
    sinh_term = torch.clamp(sinh_term, min=eps)

    # Apply the correct scaling factor
    scalar_factor = dist / sinh_term * sqrt_c

    # Return the scaled projection
    return scalar_factor * v


def half_aperture(
    x: Tensor, curv: float | Tensor = 1.0, min_radius: float = 0.1, eps: float = 1e-8
) -> Tensor:
    """
    Compute the half aperture angle of the entailment cone formed by vectors on
    the hyperboloid. The given vector would meet the apex of this cone, and the
    cone itself extends outwards to infinity.

    Args:
        x: Tensor of shape `(B, D)` giving a batch of space components of
            vectors on the hyperboloid.
        curv: Positive scalar denoting negative hyperboloid curvature.
        min_radius: Radius of a small neighborhood around vertex of the hyperboloid
            where cone aperture is left undefined. Input vectors lying inside this
            neighborhood (having smaller norm) will be projected on the boundary.
        eps: Small float number to avoid numerical instability.

    Returns:
        Tensor of shape `(B, )` giving the half-aperture of entailment cones
        formed by input vectors. Values of this tensor lie in `(0, pi/2)`.
    """

    # Ensure numerical stability in arc-sin by clamping input.
    asin_input = 2 * min_radius / ((torch.norm(x, dim=-1) * curv**0.5) + eps)
    _half_aperture = torch.asin(torch.clamp(asin_input, min=-1 + eps, max=1 - eps))

    return _half_aperture


def external_angle(
    concept: Tensor, point: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-8
):
    # print(f"x shape: {x.shape}, y shape: {y.shape}, curv: {curv}, 10 el of x : {x[:1]},\n\n 10 el of y : {y[:1]}")
    point_time = compute_time_component(point, curv)
    # print(f"x_time shape: {x_time.size()}, 10 el : {x_time[:10]}")
    concept_time = compute_time_component(concept, curv)
    # print(f"y_time shape: {y_time.size()}, 10 el : {y_time[:10]}")
    numerator = point_time + (
        (concept_time * curv * pairwise_inner(point, concept, curv))
    )
    denominator = torch.norm(concept) * torch.sqrt(
        (curv * pairwise_inner(point, concept, curv)) ** 2 - 1
    )
    angle = torch.acos(numerator / (denominator + eps))
    return angle


def oxy_angle(x: Tensor, y: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-8):
    """
    Given two vectors `x` and `y` on the hyperboloid, compute the exterior
    angle at `x` in the hyperbolic triangle `Oxy` where `O` is the origin
    of the hyperboloid.

    This expression is derived using the Hyperbolic law of cosines.

    Args:
        x: Tensor of shape `(B, D)` giving a batch of space components of
            vectors on the hyperboloid.
        y: Tensor of same shape as `x` giving another batch of vectors.
        curv: Positive scalar denoting negative hyperboloid curvature.

    Returns:
        Tensor of shape `(B, )` giving the required angle. Values of this
        tensor lie in `(0, pi)`.
    """

    # Calculate time components of inputs (multiplied with `sqrt(curv)`):
    x_time = torch.sqrt(1 / curv + torch.sum(x**2, dim=-1))
    y_time = torch.sqrt(1 / curv + torch.sum(y**2, dim=-1))

    # Calculate lorentzian inner product multiplied with curvature. We do not use
    # the `pairwise_inner` implementation to save some operations (since we only
    # need the diagonal elements).
    c_xyl = curv * (torch.sum(x * y, dim=-1) - x_time * y_time)

    # Make the numerator and denominator for input to arc-cosh, shape: (B, )
    acos_numer = y_time + c_xyl * x_time
    acos_denom = torch.sqrt(torch.clamp(c_xyl**2 - 1, min=eps))

    acos_input = acos_numer / (torch.norm(x, dim=-1) * acos_denom + eps)
    _angle = torch.acos(torch.clamp(acos_input, min=-1 + eps, max=1 - eps))

    return _angle


def oxy_angle_eval(x: Tensor, y: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-8):
    """
    Given two vectors `x` and `y` on the hyperboloid, compute the exterior
    angle at `x` in the hyperbolic triangle `Oxy` where `O` is the origin
    of the hyperboloid.

    This expression is derived using the Hyperbolic law of cosines.

    Args:
        x: Tensor of shape `(B, D)` giving a batch of space components of
            vectors on the hyperboloid.
        y: Tensor of same shape as `x` giving another batch of vectors.
        curv: Positive scalar denoting negative hyperboloid curvature.

    Returns:
        Tensor of shape `(B, )` giving the required angle. Values of this
        tensor lie in `(0, pi)`.
    """

    # Calculate time components of inputs (multiplied with `sqrt(curv)`):
    x_time = torch.sqrt(1 / curv + torch.sum(x**2, dim=-1, keepdim=True))
    y_time = torch.sqrt(1 / curv + torch.sum(y**2, dim=-1, keepdim=True))

    logger.info(f"x_time shape: {x_time.size()}")
    logger.info(f"y_time shape: {y_time.size()}")

    # Calculate lorentzian inner product multiplied with curvature. We do not use
    # the `pairwise_inner` implementation to save some operations (since we only
    # need the diagonal elements).

    # c_xyl = curv * (torch.sum(x * y, dim=-1) - x_time * y_time)
    c_xyl = curv * (y @ x.T - y_time @ x_time.T)
    logger.info(f"c_xyl shape: {c_xyl.size()}")

    # Make the numerator and denominator for input to arc-cosh, shape: (B, )
    acos_numer = y_time + c_xyl * x_time.T
    logger.info(f"acos_numer shape: {acos_numer.size()}")
    acos_denom = torch.sqrt(torch.clamp(c_xyl**2 - 1, min=eps))
    logger.info(f"acos_denom shape: {acos_denom.size()}")

    acos_input = acos_numer / (torch.norm(x, dim=-1, keepdim=True).T * acos_denom + eps)
    _angle = -torch.acos(torch.clamp(acos_input, min=-1 + eps, max=1 - eps))

    return _angle


def project_to_hyperboloid(points: Tensor, curv: float = 1.0) -> Tensor:
    """
    Project points onto the hyperboloid with given curvature.

    Args:
        points: Tensor of shape (N, D) giving space components of points
        curv: Curvature parameter (default 1.0)

    Returns:
        Tensor of shape (N, D) with points projected onto hyperboloid
    """
    # Calculate squared norm of space components
    space_norm_squared = torch.sum(points**2, dim=-1, keepdim=True)

    # Calculate what the time component would be
    time_component = torch.sqrt(1 / curv + space_norm_squared)

    # Scale the space components to satisfy the hyperboloid constraint
    scaling_factor = torch.sqrt(space_norm_squared / (time_component**2 - 1 / curv))
    projected_points = points / scaling_factor

    return projected_points


# ------------- CENTROIDS --------------- #
# give a batch of points on the hyperboloid, compute their centroid
def hyperbolic_centroid(
    points: Tensor,
    curv: float = 1.0,
    max_iter: int = 100,
    tolerance: float = 1e-6,
    verbose: bool = False,
) -> Tensor:
    """
    Compute the centroid (Fréchet mean) of points in hyperbolic space using the Lorentz model.
    This implementation properly uses the exp_map0 and log_map0 functions from hyperboloid_tools.

    Args:
        points: Tensor of shape (N, D) where N is number of points and D is dimension
        curv: Curvature parameter of hyperbolic space (default 1.0)
        max_iter: Maximum number of iterations
        tolerance: Convergence threshold
        verbose: Whether to print progress information

    Returns:
        Tensor of shape (D,) representing the centroid
    """
    # First verify and project input points if needed
    if not verify_points_on_hyperboloid(points, curv).all():
        if verbose:
            print("Warning: Input points not all on hyperboloid. Projecting...")
        points = project_to_hyperboloid(points, curv)

    # Initialize with a reasonable initial point - we'll use the origin of the tangent space
    # and then map it to the hyperboloid
    current = torch.zeros((1, points.shape[1]), device=points.device)
    current_on_manifold = exp_map0(
        current, curv
    )  # This is effectively just the origin on the hyperboloid

    # Iterative optimization using gradient descent in the tangent space
    for i in range(max_iter):
        # Compute log maps from current estimate to all points (i.e., get tangent vectors)
        log_maps = []
        for p in points:
            # Use the existing log_map0 function to get the tangent vector
            tangent_vec = log_map0(p.unsqueeze(0) - current_on_manifold, curv)
            log_maps.append(tangent_vec)

        # Stack tangent vectors and compute mean direction
        log_maps = torch.cat(log_maps, dim=0)
        mean_direction = torch.mean(log_maps, dim=0, keepdim=True)

        # Check for convergence
        movement = torch.norm(mean_direction)
        if movement < tolerance:
            if verbose:
                print(f"Converged after {i+1} iterations")
            break

        # Move in the mean direction using exponential map
        current_on_manifold = exp_map0(mean_direction, curv) + current_on_manifold

        # Ensure the point stays on the hyperboloid
        current_on_manifold = project_to_hyperboloid(current_on_manifold, curv)

        if verbose and (i + 1) % 10 == 0:
            print(f"Iteration {i+1}, movement: {movement.item():.6f}")

    if verbose and i == max_iter - 1:
        print(f"Maximum iterations ({max_iter}) reached without convergence")

    return current_on_manifold.squeeze(0)


# --------------- PARALLEL TRANSPORT --------------- #
# --------------------------------------------------------------------------------------
# Geometry-faithful displacement application:
#   Given base point `base`, sources `src` and targets `tgt`, apply at `base` the mean
#   intrinsic displacement that moves each src_i to tgt_i:
#       c = Exp_base( mean_i PT_{src_i→base}( Log_{src_i}(tgt_i) ) )
#
# Shapes (space components only, consistent with this module):
#   - base: (D,) or (B, D)
#   - src, tgt:
#       * (B, D)           -> average over B
#       * (B, N, D)        -> average over N (and over B if base is (D,))
#       * (N, D)           -> treated as (1, N, D)
#   - weights (optional): (B,N), (N,), or (B,) depending on what you want to average over
#
# Curvature:
#   - `curv` > 0 is the absolute curvature (sectional curvature = −curv).
#   - Radius R satisfies R^2 = 1/curv.
# --------------------------------------------------------------------------------------
def _lorentz_dot_full(a_full: Tensor, b_full: Tensor) -> Tensor:
    return -a_full[..., :1] * b_full[..., :1] + (a_full[..., 1:] * b_full[..., 1:]).sum(
        dim=-1, keepdim=True
    )


def _minkowski_norm_spacelike_full(v_full: Tensor, eps: float = 1e-8) -> Tensor:
    val = (v_full[..., 1:] * v_full[..., 1:]).sum(dim=-1, keepdim=True) - v_full[
        ..., :1
    ] ** 2
    return torch.sqrt(torch.clamp(val, min=eps))


def _sinhc(lam: Tensor) -> Tensor:
    return torch.where(
        torch.abs(lam) > 1e-7, torch.sinh(lam) / lam, 1.0 + (lam * lam) / 6.0
    )


def _full_from_space(p_space: Tensor, curv: float | Tensor) -> Tensor:
    t = compute_time_component(p_space, curv=curv)  # (...,1)
    return torch.cat([t, p_space], dim=-1)


def exp_map_at(
    base_space: Tensor, v_space: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-8
) -> Tensor:
    curv_t = torch.as_tensor(curv, dtype=base_space.dtype, device=base_space.device)
    sqrt_c = torch.sqrt(curv_t)
    t = compute_time_component(base_space, curv=curv_t)  # (...,1)
    xv = (base_space * v_space).sum(dim=-1, keepdim=True)  # (...,1)
    v0 = xv / (t + eps)  # (...,1)
    X_full = torch.cat([t, base_space], dim=-1)
    V_full = torch.cat([v0, v_space], dim=-1)
    v_norm = _minkowski_norm_spacelike_full(V_full, eps=eps)  # (...,1)
    lam = sqrt_c * v_norm  # (...,1)
    Y_full = torch.cosh(lam) * X_full + _sinhc(lam) * V_full
    return Y_full[..., 1:]  # space


def log_map_at(
    base_space: Tensor,
    target_space: Tensor,
    curv: float | Tensor = 1.0,
    eps: float = 1e-8,
) -> Tensor:
    curv_t = torch.as_tensor(curv, dtype=base_space.dtype, device=base_space.device)
    sqrt_c = torch.sqrt(curv_t)
    X_full = _full_from_space(base_space, curv=curv_t)
    Y_full = _full_from_space(target_space, curv=curv_t)
    inner = _lorentz_dot_full(X_full, Y_full)  # (...,1) = <X,Y>_L
    alpha = torch.clamp(-curv_t * inner, min=1.0 + eps)  # α = -<X,Y>/R^2
    Delta = torch.acosh(alpha)  # (...,1)
    U_full = Y_full + (curv_t * inner) * X_full  # (...,D+1)
    U_norm = _minkowski_norm_spacelike_full(U_full, eps=eps)  # (...,1)
    scale = torch.where(
        U_norm > 0, (Delta / (sqrt_c + 0.0)) / U_norm, torch.zeros_like(U_norm)
    )
    V_full = scale * U_full
    return V_full[..., 1:]  # space (tangent at base)


def parallel_transport_vector(
    p_space: Tensor,
    q_space: Tensor,
    v_space: Tensor,
    curv: float | Tensor = 1.0,
    eps: float = 1e-8,
) -> Tensor:
    """
    PT_{P→Q}(V) along geodesic from P to Q:
      PT(V) = V + ( <V,Q>_L / (R^2 - <P,Q>_L) ) (P + Q),   with R^2 = 1/curv
    Returns space component of the transported vector at Q.
    """
    curv_t = torch.as_tensor(curv, dtype=p_space.dtype, device=p_space.device)
    R2 = 1.0 / curv_t
    P_full = _full_from_space(p_space, curv=curv_t)
    Q_full = _full_from_space(q_space, curv=curv_t)
    tP = compute_time_component(p_space, curv=curv_t)
    pv = (p_space * v_space).sum(dim=-1, keepdim=True)
    v0 = pv / (tP + eps) # [50,1]
    V_full = torch.cat([v0, v_space], dim=-1)
    inner_PQ = _lorentz_dot_full(P_full, Q_full)  # (...,1)
    denom = R2 - inner_PQ  # (...,1)
    v_dot_Q = _lorentz_dot_full(V_full, Q_full)  # (...,1)
    coeff = v_dot_Q / (denom + eps)
    W_full = V_full + coeff * (P_full + Q_full)
    return W_full[..., 1:]  # space


def parallel_transport(
    x: Tensor,
    y: Tensor,
    curv: float | Tensor = 1.0,
    modality_alpha: float = 0.5,
    weights: Tensor | None = None,
    scale: float = 1.0,
    eps: float = 1e-8,
    return_tangent: bool = False,
    rescale: bool = True,
) -> Tensor | tuple[Tensor, Tensor]:
    """
    Apply on point x the (average) displacement defined by y with respect to the origin.

    Construction (origin-rooted):
      - For each y_i, compute d0_i = Log_0(y_i) in the tangent space at the origin.
      - Average them (optionally weighted) to get d0_avg.
      - Parallel transport to T_xH: v_x = PT_{0 → x}(d0_avg).
      - Apply at x: x' = Exp_x(scale * v_x).

    Shapes (space-only I/O, consistent with this module):
      - x: (D,) or (B_x, D)
      - y: (D,), (M, D), or (B, N, D)
      - weights (optional): (M,) or (B, N) — normalized to sum to 1

    Args:
        x: base point(s) where the displacement will be applied.
        y: point(s) defining the displacement relative to the origin.
        curv: absolute curvature (> 0), manifold has sectional curvature = −curv. Radius R^2 = 1/curv.
        weights: optional weights for averaging the per-point displacements Log_0(y_i).
        scale: scalar multiplier for the transported displacement before Exp at x.
        eps: numerical epsilon.
        return_tangent: if True, also return the averaged tangent at x (after PT).

    Returns:
        x_prime: same batch shape as x.
        If return_tangent=True, also returns v_x (the averaged tangent at x).
    """
    # Normalize x to (B_x, D)
    squeeze_out = False
    if x.dim() == 1:
        x = x.unsqueeze(0)
        squeeze_out = True
    if x.dim() != 2:
        raise ValueError("x must be (D,) or (B_x, D).")
    Bx, D = x.shape

    # Canonicalize y to (M, D)
    def _flatten_y(t: Tensor) -> Tensor:
        if t.dim() == 3:
            B, N, Dt = t.shape
            if Dt != D:
                raise ValueError(f"y last dim (D={Dt}) must match x D={D}.")
            return t.reshape(-1, D)  # (M=B*N, D)
        elif t.dim() == 2:
            if t.shape[-1] != D:
                raise ValueError(f"y last dim (D={t.shape[-1]}) must match x D={D}.")
            return t
        elif t.dim() == 1:
            if t.shape[-1] != D:
                raise ValueError(f"y last dim (D={t.shape[-1]}) must match x D={D}.")
            return t.unsqueeze(0)
        else:
            raise ValueError("y must be (D,), (M,D), or (B,N,D).")

    y_flat = _flatten_y(y)  # (M, D)
    M = y_flat.shape[0]
    modality_alpha = torch.as_tensor(modality_alpha, dtype=x.dtype, device=x.device)

    # Curvature checks : curv converted to tensor curv_t on x’s device/dtype and checked to be > 0.
    curv_t = torch.as_tensor(curv, dtype=x.dtype, device=x.device)
    if torch.any(curv_t <= 0):
        raise ValueError("curv must be > 0 (sectional curvature is −curv).")

    # ---- Helpers using this module's conventions (space-only points) ----
    def _full_from_space(p_space: Tensor) -> Tensor:
        t = compute_time_component(p_space, curv=curv_t)  # (...,1)
        return torch.cat([t, p_space], dim=-1)  # (..., D+1) as (time, space...)

    def _lorentz_dot_full(a_full: Tensor, b_full: Tensor) -> Tensor:
        return -a_full[..., :1] * b_full[..., :1] + (
            a_full[..., 1:] * b_full[..., 1:]
        ).sum(dim=-1, keepdim=True)

    def _minkowski_norm_spacelike_full(v_full: Tensor) -> Tensor:
        val = (v_full[..., 1:] * v_full[..., 1:]).sum(dim=-1, keepdim=True) - v_full[
            ..., :1
        ] ** 2
        return torch.sqrt(torch.clamp(val, min=eps))

    def _sinhc(lam: Tensor) -> Tensor:
        return torch.where(
            torch.abs(lam) > 1e-7, torch.sinh(lam) / lam, 1.0 + (lam * lam) / 6.0
        )

    def _exp_map_at(base_space: Tensor, v_space: Tensor) -> Tensor:
        # Exp_X(V) with V tangent at X enforced by v0 = (x·v)/t
        t = compute_time_component(base_space, curv=curv_t)  # (B_x,1)
        xv = (base_space * v_space).sum(dim=-1, keepdim=True)
        v0 = xv / (t + eps)
        X_full = torch.cat([t, base_space], dim=-1)
        V_full = torch.cat([v0, v_space], dim=-1)
        v_norm = _minkowski_norm_spacelike_full(V_full)
        lam = torch.sqrt(curv_t) * v_norm
        Y_full = torch.cosh(lam) * X_full + _sinhc(lam) * V_full
        return Y_full[..., 1:]

    def _pt_origin_to_x(v0_space: Tensor, x_space: Tensor) -> Tensor:
        """
        PT_{0→x}(V) with origin O=(R,0,...,0). V at origin has time=0.
        v0_space: (B_x, D), x_space: (B_x, D)
        """
        O_space = torch.zeros_like(x_space)
        O_full = _full_from_space(O_space)  # (B_x, D+1)
        X_full = _full_from_space(x_space)  # (B_x, D+1)
        V_full = torch.cat([torch.zeros_like(v0_space[..., :1]), v0_space], dim=-1)

        R2 = 1.0 / curv_t
        inner_OX = _lorentz_dot_full(O_full, X_full)  # (B_x,1)
        denom = R2 - inner_OX
        v_dot_X = _lorentz_dot_full(V_full, X_full)  # (B_x,1)
        coeff = v_dot_X / (denom + eps)
        W_full = V_full + coeff * (O_full + X_full)  # tangent at X
        return W_full[..., 1:]

    # 1) Per-point origin displacement d0_i = Log_0(y_i)
    # Prefer existing log_map0 if available; fallback to log_map_at with O
    if "log_map0" in globals():
        d0_all = log_map0(
            y_flat.to(x.dtype).to(x.device), curv=curv_t, eps=eps
        )  # (M, D)
        d0_all = d0_all / modality_alpha.exp() if rescale else d0_all
    else:
        O_space = torch.zeros((1, D), dtype=x.dtype, device=x.device)

        d0_all = log_map_at(
            O_space.expand(M, -1), y_flat.to(x.dtype).to(x.device), curv=curv_t, eps=eps
        )  # (M, D)
        d0_all = d0_all / modality_alpha.exp() if rescale else d0_all

    # 2) Average (optionally weighted)
    if M == 1 and weights is None:
        d0_avg = d0_all  # (1, D)
    else:
        if weights is not None:
            w = torch.as_tensor(weights, dtype=x.dtype, device=x.device)
            if w.dim() >= 2:
                w = w.reshape(-1)
            if w.shape[0] != M:
                raise ValueError(
                    f"weights length must match number of points M={M}, got {w.shape[0]}"
                )
            w = w / (w.sum() + eps)
            d0_avg = (w.view(M, 1) * d0_all).sum(dim=0, keepdim=True)  # (1, D)
        else:
            d0_avg = d0_all.mean(dim=0, keepdim=True)  # (1, D)

    # 3) PT d0_avg from origin to each x in the batch
    d0_avg_expand = d0_avg.expand(Bx, -1)  # (B_x, D)
    v_x = _pt_origin_to_x(d0_avg_expand, x)  # (B_x, D)

    # 4) Apply at x
    x = x * modality_alpha.exp() if rescale else x
    x_prime = _exp_map_at(x, scale * v_x)  # (B_x, D)

    if squeeze_out:
        x_prime = x_prime.squeeze(0)
        v_x = v_x.squeeze(0)

    return (x_prime, v_x) if return_tangent else x_prime


def lorentz_dot_aligned(x: Tensor, y: Tensor, curv: float | Tensor = 1.0) -> Tensor:
    """
    Batched Lorentz product <X,Y>_L for aligned batches of space-components.
    Inputs:
      x, y: (B, D) space components only
    Returns:
      (B,) with <X,Y>_L = x·y - x_time * y_time
    """
    xt = compute_time_component(x, curv=curv).squeeze(-1)  # (B,)
    yt = compute_time_component(y, curv=curv).squeeze(-1)  # (B,)
    return torch.sum(x * y, dim=-1) - xt * yt


def angle_at_p_wrt_root(
    p: Tensor,
    q: Tensor,
    curv: float | Tensor = 1.0,
    incoming: bool = False,
    eps: float = 1e-8,
) -> Tensor:
    """
    Angle at p between the geodesic p->q and the axis defined by the root and p,
    for a hyperboloid with sectional curvature -curv (curv > 0).

    Conventions:
    - p, q are (B, D) space components only; time components are recovered.
    - Root has zero spatial part and time sqrt(1/curv).

    Theory (hyperbolic law of cosines via Lorentz products):
      Let s_ab = -curv * <a,b>_L (so s_ab = cosh(d(a,b) * sqrt(curv)) >= 1).
      The angle at p between p->q and p->root (outgoing axis) is:
        cos(theta_out) = (s_pq s_po - s_qo) /
                         (sqrt(s_pq^2 - 1) sqrt(s_po^2 - 1))
      Angle wrt incoming axis (root->p at p) is:
        theta_in = pi - theta_out

    Args:
      p: (B, D) vertex space components.
      q: (B, D) target space components.
      curv: positive curvature magnitude (sectional curvature = -curv).
      incoming: if True, angle wrt incoming axis (root->p); else outgoing (p->root).
      eps: numerical epsilon.

    Returns:
      (B,) angles in radians in [0, pi].
    """
    # Time components
    p_time = compute_time_component(p, curv=curv).squeeze(-1)  # (B,)
    q_time = compute_time_component(q, curv=curv).squeeze(-1)  # (B,)

    # Root time (broadcast to batch)
    K = torch.as_tensor(curv, dtype=p.dtype, device=p.device)
    o_time = torch.sqrt(1.0 / K).expand_as(p_time)  # (B,)

    # Lorentz products
    pq_l = torch.sum(p * q, dim=-1) - p_time * q_time  # <p,q>_L
    po_l = -p_time * o_time  # <p,root>_L (root has zero spatial part)
    qo_l = -q_time * o_time  # <q,root>_L

    # s_ab = -curv * <a,b>_L  (>= 1)
    s_pq = torch.clamp(-K * pq_l, min=1.0 + eps)
    s_po = torch.clamp(-K * po_l, min=1.0 + eps)
    s_qo = torch.clamp(-K * qo_l, min=1.0 + eps)

    # Denominator
    denom = torch.sqrt(torch.clamp(s_pq * s_pq - 1.0, min=0.0)) * torch.sqrt(
        torch.clamp(s_po * s_po - 1.0, min=0.0)
    )

    # Safe cosine for outgoing angle; if denom ~ 0, define theta_out = 0
    safe = denom > eps
    cos_out = torch.ones_like(denom)
    cos_out = torch.where(safe, (s_pq * s_po - s_qo) / denom, cos_out)
    cos_out = torch.clamp(cos_out, -1.0, 1.0)

    theta_out = torch.arccos(cos_out)
    angle = torch.pi - theta_out if incoming else theta_out
    return torch.clamp(angle, min=0.0, max=float(torch.pi))


#### FRECHET MEAN/ CENTROID/ BARICENTER ####
EPS = {torch.float32: 1e-4, torch.float64: 1e-8}
TOLEPS = {torch.float32: 1e-6, torch.float64: 1e-12}

class Acosh(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        x = torch.clamp(x, min=1+EPS[x.dtype])
        z = torch.sqrt(x * x - 1)
        ctx.save_for_backward(z)
        return torch.log(x + z)

    @staticmethod
    def backward(ctx, g):
        z, = ctx.saved_tensors
        z.data.clamp(min=EPS[z.dtype])
        z = g / z
        return z, None

arcosh = Acosh.apply

def darcosh(x):
    cond = (x < 1 + 1e-7)
    x = torch.where(cond, 2 * torch.ones_like(x), x)
    x = torch.where(~cond, 2 * arcosh(x) / torch.sqrt(x**2 - 1), x)
    return x



@staticmethod
def _ldot(u, v, keepdim=False, dim=-1):
    m = u * v
    if keepdim:
        ret = torch.sum(m, dim=dim, keepdim=True) - 2 * m[..., 0:1]
    else:
        ret = torch.sum(m, dim=dim, keepdim=False) - 2 * m[..., 0]
    return ret

def frechet_hyperboloid_forward(X, w, K=-1.0, max_iter=1000, rtol=1e-6, atol=1e-6, verbose=False):
    """
    Args
    ----
        X (tensor): point of shape [..., points, dim]
        w (tensor): weights of shape [..., points]
        K (float): curvature (must be negative)
    Returns
    -------
        frechet mean (tensor): shape [..., dim]
    """
    mu = X[..., 0, :].clone()

    mu_prev = mu
    iters = 0
    for _ in range(max_iter):
        inner = K * _ldot(X, mu.unsqueeze(-2), keepdim=True)# ; print(f"inner is {inner}")
        u = (w.unsqueeze(-1) * darcosh(inner) * X).sum(dim=-2)# ; print(f"u is {u}")
        mu = u / (K * _ldot(u, u, keepdim=True)).sqrt()# ; print(f"mu is {mu}")

        dist = (mu - mu_prev).norm(dim=-1)
        prev_dist = mu_prev.norm(dim=-1)

        if (dist < atol).all() or (dist / prev_dist < rtol).all():
            break

        mu_prev = mu
        iters += 1

    if verbose:
        print(iters)

    return mu