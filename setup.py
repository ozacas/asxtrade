import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

TEST_REQUIREMENTS = [
    'pytest',
    'pytest-django',
    'pylint',
    'pylint_django',
    'git-pylint-commit-hook',
]

setuptools.setup(
    name="ozacas", 
    version="0.0.1",
    author="ozacas",
    author_email="https://github.com/ozacas",
    description="Download ASX research data and ingest into MongoDB. Viewer application in Django",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ozacas/asxtrade",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
    tests_require=TEST_REQUIREMENTS,
    package_dir={'': 'src'},  
)

