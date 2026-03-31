# ---------------------------------------
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# ---------------------------------------

"""
Visualization utilities for hyperbolic geometry in the Lorentz model.
Provides functions to plot entailment cones and points in 2D projections.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
import matplotlib.patches as mpatches
from typing import Optional, Tuple, List
import sys

sys.path.append(".")
import hycoclip.lorentz as L


def poincare_projection(points: torch.Tensor, curv: float = 1.0) -> np.ndarray:
    """
    Project points from Lorentz model to Poincaré disk for visualization.
    
    Args:
        points: Tensor of shape (N, D) with space components
        curv: Curvature parameter
        
    Returns:
        numpy array of shape (N, 2) with 2D Poincaré coordinates
    """
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
    
    # Take first 2 dimensions for visualization
    if points.shape[-1] > 2:
        points_2d = points[..., :2]
    else:
        points_2d = points
    
    # Compute time component
    time_comp = np.sqrt(1 / curv + np.sum(points_2d**2, axis=-1, keepdims=True))
    
    # Poincaré projection: x_poincare = x_space / (1 + x_time)
    poincare = points_2d / (1 + time_comp)
    
    return poincare


def plot_entailment_cone_2d(
    apex: torch.Tensor,
    points: Optional[List[Tuple[torch.Tensor, str, str]]] = None,
    curv: float = 1.0,
    min_radius: float = 0.1,
    figsize: Tuple[int, int] = (10, 10),
    title: str = "Entailment Cone in Poincaré Disk",
    save_path: Optional[str] = None,
    show_grid: bool = True,
) -> plt.Figure:
    """
    Plot entailment cone and points in 2D Poincaré disk projection.
    
    Args:
        apex: Tensor of shape (D,) representing the apex of the cone
        points: List of tuples (point_tensor, label, color) to plot
        curv: Curvature parameter
        min_radius: Minimum radius for half-aperture calculation
        figsize: Figure size
        title: Plot title
        save_path: If provided, save figure to this path
        show_grid: Whether to show grid
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Draw Poincaré disk boundary
    circle = plt.Circle((0, 0), 1, color='black', fill=False, linewidth=2)
    ax.add_patch(circle)
    
    # Project apex to Poincaré disk
    apex_poincare = poincare_projection(apex.unsqueeze(0) if apex.dim() == 1 else apex, curv)[0]
    
    # Compute half-aperture
    half_aperture_angle = L.half_aperture(
        apex.unsqueeze(0) if apex.dim() == 1 else apex, 
        curv, 
        min_radius
    ).item()
    
    # Draw the cone
    # In Poincaré disk, geodesics from origin are straight lines
    # The cone is a wedge centered at origin
    
    # Compute angle of apex from origin
    apex_angle = np.arctan2(apex_poincare[1], apex_poincare[0])
    apex_angle_deg = np.degrees(apex_angle)
    half_aperture_deg = np.degrees(half_aperture_angle)
    
    # Draw cone as a wedge
    wedge = Wedge(
        (0, 0), 1.0, 
        apex_angle_deg - half_aperture_deg,
        apex_angle_deg + half_aperture_deg,
        alpha=0.3, 
        color='lightblue',
        label='Entailment Cone'
    )
    ax.add_patch(wedge)
    
    # Draw cone boundaries (geodesics from origin)
    boundary_angle_1 = apex_angle - half_aperture_angle
    boundary_angle_2 = apex_angle + half_aperture_angle
    
    ax.plot(
        [0, np.cos(boundary_angle_1)], 
        [0, np.sin(boundary_angle_1)],
        'b--', linewidth=1.5, alpha=0.7, label='Cone Boundary'
    )
    ax.plot(
        [0, np.cos(boundary_angle_2)], 
        [0, np.sin(boundary_angle_2)],
        'b--', linewidth=1.5, alpha=0.7
    )
    
    # Plot origin
    ax.plot(0, 0, 'ko', markersize=8, label='Origin', zorder=5)
    
    # Plot apex
    ax.plot(
        apex_poincare[0], apex_poincare[1], 
        'r^', markersize=12, label='Apex (Concept)', zorder=5
    )
    
    # Draw line from origin to apex
    ax.plot(
        [0, apex_poincare[0]], 
        [0, apex_poincare[1]],
        'r-', linewidth=2, alpha=0.5, label='Cone Axis'
    )
    
    # Plot additional points if provided
    if points is not None:
        for point, label, color in points:
            point_poincare = poincare_projection(
                point.unsqueeze(0) if point.dim() == 1 else point, 
                curv
            )[0]
            
            # Check if point is in cone
            is_in_cone = L.is_in_entailment_cone(apex, point, curv, min_radius)
            marker = 'o' if is_in_cone else 'x'
            markersize = 10 if is_in_cone else 12
            
            ax.plot(
                point_poincare[0], point_poincare[1],
                marker=marker, color=color, markersize=markersize,
                label=f'{label} ({"inside" if is_in_cone else "outside"})',
                zorder=5
            )
            
            # Draw line from origin to point
            ax.plot(
                [0, point_poincare[0]], 
                [0, point_poincare[1]],
                color=color, linewidth=1, alpha=0.3, linestyle=':'
            )
    
    # Formatting
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal')
    ax.set_xlabel('x₁', fontsize=12)
    ax.set_ylabel('x₂', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    if show_grid:
        ax.grid(True, alpha=0.3)
    
    # Add legend
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    
    # Add text annotation with cone info
    info_text = f'Half-aperture: {half_aperture_deg:.2f}° ({half_aperture_angle:.4f} rad)'
    ax.text(
        0.02, 0.98, info_text,
        transform=ax.transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
        fontsize=10
    )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig


def plot_multiple_cones(
    cones: List[Tuple[torch.Tensor, str, str]],
    points: Optional[List[Tuple[torch.Tensor, str, str]]] = None,
    curv: float = 1.0,
    min_radius: float = 0.1,
    figsize: Tuple[int, int] = (10, 10),
    title: str = "Multiple Entailment Cones",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot multiple entailment cones in the same Poincaré disk.
    
    Args:
        cones: List of tuples (apex_tensor, label, color)
        points: List of tuples (point_tensor, label, color) to plot
        curv: Curvature parameter
        min_radius: Minimum radius for half-aperture calculation
        figsize: Figure size
        title: Plot title
        save_path: If provided, save figure to this path
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Draw Poincaré disk boundary
    circle = plt.Circle((0, 0), 1, color='black', fill=False, linewidth=2)
    ax.add_patch(circle)
    
    # Plot origin
    ax.plot(0, 0, 'ko', markersize=8, label='Origin', zorder=5)
    
    # Plot each cone
    colors_used = []
    for apex, label, color in cones:
        apex_poincare = poincare_projection(
            apex.unsqueeze(0) if apex.dim() == 1 else apex, 
            curv
        )[0]
        
        half_aperture_angle = L.half_aperture(
            apex.unsqueeze(0) if apex.dim() == 1 else apex,
            curv,
            min_radius
        ).item()
        
        apex_angle = np.arctan2(apex_poincare[1], apex_poincare[0])
        apex_angle_deg = np.degrees(apex_angle)
        half_aperture_deg = np.degrees(half_aperture_angle)
        
        # Draw cone
        wedge = Wedge(
            (0, 0), 1.0,
            apex_angle_deg - half_aperture_deg,
            apex_angle_deg + half_aperture_deg,
            alpha=0.2,
            color=color,
            label=f'{label} Cone'
        )
        ax.add_patch(wedge)
        
        # Draw boundaries
        boundary_angle_1 = apex_angle - half_aperture_angle
        boundary_angle_2 = apex_angle + half_aperture_angle
        
        ax.plot(
            [0, np.cos(boundary_angle_1)],
            [0, np.sin(boundary_angle_1)],
            color=color, linestyle='--', linewidth=1.5, alpha=0.7
        )
        ax.plot(
            [0, np.cos(boundary_angle_2)],
            [0, np.sin(boundary_angle_2)],
            color=color, linestyle='--', linewidth=1.5, alpha=0.7
        )
        
        # Plot apex
        ax.plot(
            apex_poincare[0], apex_poincare[1],
            '^', color=color, markersize=12, label=f'{label} Apex', zorder=5
        )
        
        colors_used.append(color)
    
    # Plot additional points
    if points is not None:
        for point, label, color in points:
            point_poincare = poincare_projection(
                point.unsqueeze(0) if point.dim() == 1 else point,
                curv
            )[0]
            
            ax.plot(
                point_poincare[0], point_poincare[1],
                'o', color=color, markersize=10, label=label, zorder=5
            )
    
    # Formatting
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal')
    ax.set_xlabel('x₁', fontsize=12)
    ax.set_ylabel('x₂', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig