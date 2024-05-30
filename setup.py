# Standard Library
import os
from typing import List

from setuptools import find_packages, setup

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))


def load_requirements(filename: str) -> List[str]:
    with open(os.path.join(PROJECT_ROOT, filename), "r") as f:
        return f.read().splitlines()

setup(
    name="simple_classifier",
    setup_requires=["setuptools_scm", "pytest-runner"],
    use_scm_version={"fallback_version": "no_git"},
    description="Image classification train",
    author="Aleksandr Kalashnikov",
    packages=find_packages(exclude=("mleco", "tests")),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3.9",
    ],
    install_requires=load_requirements("requirements/requirements.txt"),
    extras_require={
        "format": load_requirements("requirements/requirements-format.txt"),
        "lint": load_requirements("requirements/requirements-lint.txt"),
        "test": load_requirements("requirements/requirements-test.txt"),
    },
    tests_require=["pytest-cov"],
    python_requires=">=3.9",
)
