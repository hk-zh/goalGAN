# Easy Environment
### Installation
Please install:
* Pytorch
* Stable-Baselines

Install the system dependencies:
```bash
cat sys-packages.txt | xargs sudo apt install -y
```
And the Python dependencies:
```bash
python -m pip install -r requirements.txt
```
As well as this package:
```bash
python -m pip install -e .
```
### Run
Requires python 3.6+
```bash
python main.py --seed=10
```
# Testing
This Python package must be installed. Then run the tests. Disable the Tensorflow warnings.
```bash
python -m pytest --disable-warnings
```