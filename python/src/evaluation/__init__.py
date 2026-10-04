"""
src/evaluation/__init__.py
Convenience imports for the evaluation package.
"""

from .h36m_loader import parse_h36m_file, iter_h36m_dataset, GTAngles
from .metrics import (
    evaluate_framework,
    compute_joint_metrics,
    print_metrics_table,
    metrics_to_dict,
    JOINTS,
    JOINTS_LEFT,
    JOINTS_BILATERAL,
    JOINT_LABELS,
    FrameworkMetrics,
    JointMetrics,
)
from .statistics import (
    compare_systems,
    comparison_report_to_dicts,
    paired_t_test,
    wilcoxon_test,
    cohens_d_paired,
    bootstrap_ci,
)
from .protocol import (
    H36M_SPLITS,
    H36M_ALL_SUBJECTS,
    get_split,
    loso_folds,
    assert_no_leakage,
    describe_protocol,
)
from .ablation import run_filter_ablation
from .report_export import (
    build_metadata,
    export_excel,
    export_frame_level_csv,
    export_metadata_json,
)
