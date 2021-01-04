from setuptools import setup, find_packages


with open("requirements.txt") as f:
    requirements = f.read().strip().split("\n")

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()


setup(
    name="dodola",
    version="0.1.0a0",
    description="GCM bias-correction and downscaling.",
    long_description=readme + "\n\n" + history,
    long_description_content_type="text/x-rst",
    url="https://github.com/ClimateImpactLab/dodola",
    license="MIT license",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Natural Language :: English",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
    ],
    project_urls={
        "Source": "https://github.com/ClimateImpactLab/dodola",
        "Tracker": "https://github.com/ClimateImpactLab/dodola/issues",
    },
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    python_requires=">=3.7",
    # install_requires=requirements,
    entry_points="""
    [console_scripts]
    dodola=dodola.cli:dodola_cli
""",
)
