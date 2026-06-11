def test_imports_smoke():
    """STEP 2-4 modules import senza errori."""
    import connectivity.fc_dispatcher  # noqa: F401
    import parcellation.extract_label_tc  # noqa: F401
    import source_reconstruction.finalize_inverse  # noqa: F401
