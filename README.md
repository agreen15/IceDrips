# IceDrips

Readme: Software package for "Dripping to Destruction" by Green and Cooper (2022)--in review.

Engine: Underworld 2, Version 2.11.0 (underworld2/2.11.0)

Underworld 2 script is meant to be read through the "Jupyter Notebooks" python notebook platform. All .ipynb files found in here may be read through that platform or similar notebook software. 

The python notebooks allow for clearer code organization and reading, as well as image previews of certain figures and data. However, standard python scripts (".py") of each code file are also provided if the user does not have the capability of viewing the original notebook files.

"IceDripMasterScript.ipynb" represents the core modeling software.  Enclosed is all the code necessary to run the model and obtain raw results found in Green & Cooper (2022).  

The intended use of the master script is to change flags and variables located near the beginning of the script (in cells 8-11) and batch export as individually named .py  files to run through some HPC system. For example: export the master script as "IceDripB1T1.py" for the first model run of the first batch, "IceDripB1T2" for the second model run of the first batch, etc...

Also included is two figure generating scripts ("FigureGenerator1.ipynb" and "FigureGenerator2.ipynb") used to make the two major results figures found in Green & Cooper (2022). These scripts require collated results collected from several model runs.

Sample outputs from the figure generating scripts are also enclosed as "Figure1.png" and "Figure2.png", respectively. 
