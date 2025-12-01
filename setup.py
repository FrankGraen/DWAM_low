import itertools

from setuptools import find_packages, setup  # noqa: F401

INSTALL_REQUIRES = [
    # generic
    "numpy",
    "torch",
    "prettytable==3.3.0",
    "pymeshlab",
    "open3d",
    # devices
    "hidapi",
    "skrl==1.4.1",
    "wandb",
    "opencv-python",
    "isaaclab==2.2.0"
]

# url=EXTENSION_TOML_DATA["package"]["repository"], # add later
# version=EXTENSION_TOML_DATA["package"]["version"],
# description=EXTENSION_TOML_DATA["package"]["description"],
# keywords=EXTENSION_TOML_DATA["package"]["keywords"],
EXTRAS_REQUIRE = {
    # "rsl_rl": ["rsl_rl@git+https://github.com/leggedrobotics/rsl_rl.git"],
}

# cumulation of all extra-requires
EXTRAS_REQUIRE["all"] = list(itertools.chain.from_iterable(EXTRAS_REQUIRE.values()))
setup(
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    packages=["envs"],
    classifiers=["Natural Language :: English", "Programming Language :: Python :: 3.10"],
    zip_safe=False,
)
