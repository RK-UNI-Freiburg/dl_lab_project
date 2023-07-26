# Deep Learning Lab Project

The goal of the project **Transformers for EEG Data** is to assess and analyze the performance of Transformers when 
applied to EEG data. Furthermore, the goal of this project is also to apply ProbTransformers on the same data to 
handle sequential data better. As an extension of the project, it is also expected to apply modern-day Hyperparameter 
Optimization techniques like Hyperband and BOHB such that we get the best model with the best hyperparameters yielding 
the best possible performance on this dataset.

For more information you can refer to this 
[Google Doc](https://docs.google.com/document/d/1N7uG7VsaE7LVqsaoxjrBQCio4SGWjhkmxrTy3-VaLP8/edit?usp=sharing) here.

### Cloning the Repository and Installing Dependencies
Please following the below instructions to clone this repository and install the requirements.

- Open a terminal and move to your workspace.
- Run `git clone https://github.com/IndrashisDas/dl_lab_project.git`
- Run `cd dl_lab_project`
- Create a virtual env by following the below commands,
  - Open Anaconda Prompt
  - Run `conda create --name dllabproject python=3.9 -y`
  - Run `conda activate dllabproject`
- Set up a Python Interpreter in PyCharm
  - Open the dl_lab_project repository via PyCharm
  - Click `Files > Settings`
  - Select `Project: dl_lab_project`
  - Click on `Python Interpreter > Add Interpreter > Add Local Interpreter > Conda Environment`
  - Select `dllabproject` under `Use existing environment` 
  - Apply the settings
- Open a terminal in PyCharm - the conda virtual environment should be activated now
- Run `pip install -r requirements.txt`
- Optional (for the contributors) - If you install any new python libraries, then please update the 
`requirements.txt` file by running the command `pip freeze > requirements.txt`. Further, push this new requirements 
file to the Git repository.

### About the Dataset and Generating Summary Statistics

We use the MOABBDataset - BNCI2014001 here. You can read about this more at this 
[link](http://moabb.neurotechx.com/docs/generated/moabb.datasets.BNCI2014001.html) here. Further, you can run the 
following command from command prompt when located in the root folder of the project to generate the summary 
statistics of this dataset.

`python -m src.data_summary_stats -d "BNCI2014001" -s "1,2,3,4,5,6,7,8,9" -dfn "data" -lf 4.0 -hf 38.0 -emsf 1e-3 -ibs 1000 -tsos -0.5 -tss 0.8 -nc 4 -sspp "plots/summary_statistics"`

### Implementations and Execution

- The transformer architectures have been designed in the ./src/networks.py file. Further the ./src/training.py file is a 
dynamic training pipeline which chooses the architecture from ./src/networks.py based on the command line arguments. 
- The training commands for every run can be found in the ./training_commands.txt file.
- The loss and accuracy plots for every run can be found in the ./plots/training_results folder. The plots consist of 
two subplots each having the training and validation loss and training and validation accuracy respectively. 
- In the following table you can find the implementations and their respective loss and accuracy. The Run ID field 
corresponds to the training commands in the ./training_commands.txt file.

| Run ID | Training Loss | Training Accuracy | Validation Loss | Validation Accuracy | Evaluation Loss | Evaluation Accuracy |
|:------:|:-------------:|:-----------------:|:---------------:|:-------------------:|:---------------:|:-------------------:|
|   1    |               |                   |                 |                     |                 |                     |
|   2    |               |                   |                 |                     |                 |                     |
|   3    |               |                   |                 |                     |                 |                     |
|   4    |               |                   |                 |                     |                 |                     |
|   5    |               |                   |                 |                     |                 |                     |
|   6    |               |                   |                 |                     |                 |                     |
|   7    |               |                   |                 |                     |                 |                     |
|   8    |               |                   |                 |                     |                 |                     |
|   9    |               |                   |                 |                     |                 |                     |
|   10   |               |                   |                 |                     |                 |                     |
|   11   |               |                   |                 |                     |                 |                     |
|   12   |               |                   |                 |                     |                 |                     |
|   13   |               |                   |                 |                     |                 |                     |
|   14   |               |                   |                 |                     |                 |                     |
|   15   |               |                   |                 |                     |                 |                     |
|   16   |               |                   |                 |                     |                 |                     |
|   17   |               |                   |                 |                     |                 |                     |
|   18   |               |                   |                 |                     |                 |                     |
|   19   |               |                   |                 |                     |                 |                     |
|   20   |               |                   |                 |                     |                 |                     |

