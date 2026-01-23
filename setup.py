from setuptools import setup, find_packages

setup(
    name="deepcytoftool",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas",code e
        "numpy",
        "scikit-learn"
    ],
    entry_points={
        "console_scripts": [
            "deepcytoftool=module.entrypoint_deepcytof:main"
        ]
    },
)
