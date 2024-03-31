import importlib

REQUIRED_PACKAGES = ["plotly", "dash", "dash_bootstrap_components"]


def _check_dependencies():
    missing_packages = []
    for package in REQUIRED_PACKAGES:
        try:
            importlib.import_module(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        message = (
            f"The 'pymotion.render.viewer' subpackage requires the following packages to be installed: "
            f"{', '.join(missing_packages)} \n"
            f"Please install them using pip: \n"
            f"pip install upc-pymotion[viewer] \n"
        )
        raise ImportError(message)


# Call the check when the subpackage is imported
_check_dependencies()
