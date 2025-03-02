# LegalCore

## Overview

This repository contains the code and datasets used in the paper:

> **"LegalCore: A Dataset for Event Coreference Resolution in Legal Documents"**\
> *Authors: Kangda Wei, Xi Shi, Jonathan Tong, Sai Ramana Reddy, Anandhavelu Natarajan, Rajiv Jain, Aparna Garimella, Ruihong Huang*

## Installation

To get started with LegalCore, clone the repository and install the required dependencies:

```bash
git clone https://github.com/WeiKangDa/LegalCore.git
cd LegalCore
pip install -r requirements.txt
```

**Note**: Ensure that you have Python 3.6 or higher installed.

## Usage

To run LLMs evaluation, run the commands in ```./script/batch_run.slurm```, modify the parameters based on your need.

To run the supervised baseline, run the commands in ```./script/supervised_detection.slurm``` for the event detection baseline. Run the commands in ```./script/supervised_coreference.slurm``` for the event coreference baseline.

## Project Structure

The repository is organized as follows:

```
LegalCore/
│
├── baseline/           # Core functionalities
│   ├── __init__.py
│   └── ...
│
├── data/              # Raw and processed data
│   ├── raw_data/
│   └── ...
│
├── post_processing/   # Post-processing utilities
│   ├── __init__.py
│   └── ...
│
├── pre_processing/    # Pre-processing tools
│   ├── __init__.py
│   └── ...
│
├── script/            # Example scripts
│   └── ...
│
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt  # Python dependencies
```

## Citation

If you use this work, please cite:

```bibtex
@misc{wei2025legalcoredatasetlegaldocuments,
      title={LegalCore: A Dataset for Legal Documents Event Coreference Resolution}, 
      author={Kangda Wei and Xi Shi and Jonathan Tong and Sai Ramana Reddy and Anandhavelu Natarajan and Rajiv Jain and Aparna Garimella and Ruihong Huang},
      year={2025},
      eprint={2502.12509},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.12509}, 
}
```

## License

LegalCore is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details. 

---

For questions, please open an issue or contact kangda@tamu.edu.