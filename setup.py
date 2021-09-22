r"""Setup File - replace with toml once PEP 660 has widespread adoption."""
import io
import os
import re

import setuptools

NAME = "linodenet"

with open(f"{NAME}/VERSION", "r") as file:
    VERSION = file.read()

if "CI_PIPELINE_IID" in os.environ:
    BUILD_NUMBER = os.environ["CI_PIPELINE_IID"]
    VERSION += f".{BUILD_NUMBER}"


def _read(filename):
    filename = os.path.join(os.path.dirname(__file__), filename)
    text_type = type("")
    with io.open(filename, mode="r", encoding="utf-8") as fd:
        return re.sub(text_type(r":[a-z]+:`~?(.*?)`"), text_type(r"``\1``"), fd.read())


setuptools.setup(
    name=NAME,
    version=VERSION,
    url="https://git.tu-berlin.de/bvt-htbd/kiwi/tf1/linodenet",
    license="MIT",
    author="Randolf Scholz",
    author_email="scholz@ismll.uni-hildesheim.de",
    description="Linear ODE Network for Time Series Forecasting",
    long_description=_read("README.rst"),
    long_description_content_type="test/X-rst",
    packages=setuptools.find_packages(
        exclude="test"
    ),  # include all packages other than test
    install_requires=[
        "scipy",
        "torch>=1.9.0",
    ],
    tests_require=[
        "scipy",
        "numpy>=1.21",
        "matplotlib",
        "torch>=1.9.0",
        "tqdm",
    ],
    # include_package_data=True,  # <-- This MUST NOT be set https://stackoverflow.com/a/23936405/9318372  # noqa
    package_data={
        #
    },
    exclude_package_data={"": ["virtualenv.yaml"]},
)
