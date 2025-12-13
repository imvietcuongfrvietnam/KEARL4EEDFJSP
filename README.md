KEARL for EEDFJSP Implementation
This repository contains the implementation of the paper: "A knowledge-guided evolutionary algorithm incorporating reinforcement learning for energy efficient dynamic flexible job shop scheduling problem with machine breakdowns".

🚀 How to use
Follow these steps to set up the environment and run the algorithm.

1. Clone the repository
Clone the code to your local machine and navigate into the directory:

Bash

git clone https://github.com/imvietcuongfrvietnam/KEARL4EEDFJSP.git
cd KEARL4EEDFJSP
2. Install Miniconda
If you haven't installed Conda yet, download and install Miniconda here: Miniconda Installation Guide

3. Create a new environment
Create a virtual environment (we recommend using Python 3.8 or higher):

Bash

conda create -n kearl4eedfjsp python=3.9
4. Activate the environment
Activate the newly created environment:

Bash

conda activate kearl4eedfjsp
5. Install dependencies
Install the required libraries using pip:

Bash

pip install -r requirements.txt
(Note: Use -r flag to install from a file)

6. Run the code
Start the algorithm by running the main script:

Bash

python main.py
📂 Dataset
The benchmark dataset used in this project can be downloaded from the link below:

Download Link: Baidu Cloud (Pan.baidu.com)

Extraction Code: yzxx

Note: After downloading, please extract the files and place them into the data/ folder in the root directory of this project so the DataLoader can find them.
