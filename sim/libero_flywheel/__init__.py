"""LIBERO simulation flywheel — VLM-guided plan-and-execute over LIBERO tasks.

Entry point: ``vlm_feedback_flywheel.py`` (unified sim runner; choose task via
``--args.task lego|drawer``). The flywheel logic lives in the ``vlm_flywheel``
subpackage; dataset post-processing (dense primitive labeling) lives in
``data_processing``.
"""
