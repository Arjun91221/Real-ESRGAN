
# Real-ESRGAN


## Installation

Follow these steps to set up the project

1. Clone the repository

```bash
git clone https://github.com/Arjun91221/Real-ESRGAN.git
cd Real-ESRGAN
```

2. Create a virtual environment

```bash
python3 -m venv venv
source venv\bin\activate
```

3. Install the required packages

```bash
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
python setup.py develop
```

4. Download Weights

```bash
cd weights
wget https://huggingface.co/lllyasviel/Annotators/resolve/main/RealESRGAN_x4plus.pth
cd ..
```

## Running the Application

You can run the application using one of the following commands

1. Using Python

```bash
  python main.py
```

2. Using Uvicorn

```bash
  uvicorn main:app --reload --host 0.0.0.0 --port 8001
```


## Usage

After running the application, you can use the provided endpoints to process images and detect hands. Ensure you have the necessary input images and follow the API documentation for more details on how to use the endpoints effectively.


## Acknowledgements

Special thanks to the developers and contributors of Real-ESRGAN and other open-source libraries used in this project.

