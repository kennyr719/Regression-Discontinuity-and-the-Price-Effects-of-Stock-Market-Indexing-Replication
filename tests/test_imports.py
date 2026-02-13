"""Smoke tests to verify all modules can be imported."""


def test_import_tables():
    """Test that tables module can be imported."""
    from auxiliary import tables

    assert hasattr(tables, "create_table1")


def test_import_predictions():
    """Test that predictions module can be imported."""
    from auxiliary import predictions

    assert hasattr(predictions, "prepare_data")


def test_import_plots():
    """Test that plots module can be imported."""
    from auxiliary import plots

    assert hasattr(plots, "plot_figure1")
