# dl_lab_project

The goal of the project Transformers for EEG Data is to assess and analyze the performance of Transformers when applied to EEG data. Furthermore, the goal of this project is also to apply ProbTransformers on the same data to handle sequential data better. As an extension of the project, it is also expected to apply modern-day Hyperparameter Optimization techniques like Hyperband and BOHB such that we get the best model with the best hyperparameters yielding the best performance on this dataset.

For more information you can refer to this [Google Doc](https://docs.google.com/document/d/1N7uG7VsaE7LVqsaoxjrBQCio4SGWjhkmxrTy3-VaLP8/edit?usp=sharing) here.

### Cloning the Repository and Install Dependencies
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
- Open a terminal in PyCharm - the conda virtual environment should be activated
- Run `pip install -r requirements.txt`
- Optional (for the contributors) - If you install any new python libraries, then please update the `requirements.txt` file by running the command `pip freeze > requirements.txt`

### About the Dataset and Generating Summary Statistics

We use the MOABBDataset - BNCI2014001 here. You can read about this more at this [link](http://moabb.neurotechx.com/docs/generated/moabb.datasets.BNCI2014001.html) here. Further, you can run the following command from command prompt when located in the root folder of the project to generate the summary statistics of this dataset.

`python -m src.data_summary_stats -d "BNCI2014001" -s "1,2,3,4,5,6,7,8,9" -dfn "data"`