"""Phase 0 smoke test: the package imports and the demo entry point runs."""

import feathersim


def test_package_imports_and_has_version():
    assert isinstance(feathersim.__version__, str)
    assert feathersim.__version__  # non-empty


def test_demo_runs(capsys):
    from feathersim import demo

    demo.main()
    out = capsys.readouterr().out
    assert "FeatherSim" in out
