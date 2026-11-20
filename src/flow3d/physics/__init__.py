"""3D discrete differential operators and real boundary-condition diagnostics."""

from .boundary import BoundaryDiagnostics, evaluate_cylinder_boundary_conditions, first_fluid_shell_mask
from .operators import (
    build_3d_derivative_mask,
    centered_difference_3d,
    curl_h,
    div_h,
    valid_div_of_curl_mask,
    verify_discrete_div_curl_identity,
)

__all__ = [
    "BoundaryDiagnostics",
    "build_3d_derivative_mask",
    "centered_difference_3d",
    "curl_h",
    "div_h",
    "evaluate_cylinder_boundary_conditions",
    "first_fluid_shell_mask",
    "valid_div_of_curl_mask",
    "verify_discrete_div_curl_identity",
]
