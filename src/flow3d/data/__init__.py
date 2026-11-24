"""Data schema, normalization, inspection, adapters, and splits for FlowWorld-3D V1.3."""

from .adapters import (
    TrajectorySegmentRecord,
    detect_temporal_rank_degeneracy,
    inspect_and_segment_c04_shedding_directory,
    inspect_world_model_dataset_manifest,
    load_c04_benchmark_npz,
    load_world_model_h5_trajectory,
)
from .dataset import Flow3DFrameSample, Flow3DH5Dataset, collate_flow3d_stage1, create_stage1_dataloaders
from .normalization import Flow3DNormalizer, Flow3DNormStats
from .schema import (
    DatasetSchema3D,
    convert_h5_6channel_to_field_and_geometry,
    denormalize_star_to_si,
    load_yaml_or_json_config,
    normalize_si_to_star,
    transpose_yzx_to_zyx_fields,
)
from .splits import SplitAssignment, group_split_by_provenance, temporal_block_split_with_purge

__all__ = [
    "DatasetSchema3D",
    "Flow3DFrameSample",
    "Flow3DH5Dataset",
    "Flow3DNormStats",
    "Flow3DNormalizer",
    "SplitAssignment",
    "TrajectorySegmentRecord",
    "collate_flow3d_stage1",
    "convert_h5_6channel_to_field_and_geometry",
    "create_stage1_dataloaders",
    "denormalize_star_to_si",
    "detect_temporal_rank_degeneracy",
    "group_split_by_provenance",
    "inspect_and_segment_c04_shedding_directory",
    "inspect_world_model_dataset_manifest",
    "load_c04_benchmark_npz",
    "load_world_model_h5_trajectory",
    "load_yaml_or_json_config",
    "normalize_si_to_star",
    "temporal_block_split_with_purge",
    "transpose_yzx_to_zyx_fields",
]
