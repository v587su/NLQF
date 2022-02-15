import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="nlqf",
    version="0.1.13",
    author="Zhensu Sun",
    author_email="zhensuuu@gmail.com",
    description="A tool for fittering code comments",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    install_requires=['torch==1.3.1','numpy','sklearn','nltk'], 
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)