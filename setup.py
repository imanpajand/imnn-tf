import setuptools

with open("README.rst", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="imnn-tf",
    version="0.2.0",
    author="Tom Charnock",
    author_email="tom@charnock.fr",
    description="Using neural networks to extract sufficient statistics from \
        data by maximising the Fisher information",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://bitbucket.org/tomcharnock/imnn-tf",
    packages=["imnn_tf", "imnn_tf.lfi", "imnn_tf.utils"],
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3",
    install_requires=[
          "tensorflow>=2.1.0",
          "tqdm>=4.31.0",
          "numpy>=1.16.0",
          "scipy>=1.4.1",
          "matplotlib"],
    include_package_data=True
)
