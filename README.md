
## Installation & Setup

### 1. Install Git

**Windows:**
Download and install from [git-scm.com](https://git-scm.com/install/)

### 2. Clone the Repository

Open your terminal/command prompt and run:

```bash
git clone https://github.com/tusineav/torch_mnist
```

### 3. Install Python

Download from [python.org](https://www.python.org/downloads/)
The most up-to-date Python should be fine, but I use 3.11.8

### 4. Create a Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS & Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

Your terminal prompt should show `(venv)` at the beginning. If it doesn't, your virtual environment is not active.

### 5. Install Python libraries

With your virtual environment activated, install the required libraries:

```bash
pip install -r requirements.txt
```

requirements.txt was created based on my environment on Windows 11.

Depending on your operating system and environment this probably won't work. If not, any modern version of the dependencies should work fine.

## Run the train script

```bash
python train.py -m cnn
```
