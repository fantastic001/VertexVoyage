# Installation and development setup

## Installation 

To install VertexVoyage using pip, you can follow these steps:

1. Open your terminal or command prompt.

2. Make sure you have pip installed. You can check by running the command `pip --version`. If pip is not installed, you can install it by following the instructions at https://pip.pypa.io/en/stable/installing/.

3. Once you have pip installed, you can use the following command to install VertexVoyage:

   ```
   pip install vertex_voyage
   ```

   This command will download and install the latest version of VertexVoyage from the Python Package Index (PyPI).

   Note: Make sure you have an active internet connection for pip to download the package.

To install VertexVoyage using the setup.py file, you can follow these steps:

1. Download the VertexVoyage source code from the repository or obtain the setup.py file.

2. Open your terminal or command prompt and navigate to the directory where the setup.py file is located.

3. Run the following command to install VertexVoyage:

   ```
   python setup.py install
   ```

   This command will execute the setup.py file and install VertexVoyage on your system.

   Note: Make sure you have Python installed and added to your system's PATH environment variable.

Both methods will install VertexVoyage and its dependencies on your system, allowing you to use it in your cluster.


## Development 

### Conda setup

Create environment

```
conda create myenv -c conda-forge rust python=3.9 
```

Activate conda and install dependencies:

```
conda activate myenv
pip install -r requirements.txt
```

Compile Rust code with `maturin`

```
cd vertex_voyage_native
maturin develop
```

