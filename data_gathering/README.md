# Data_Gathering_Tools

Tools to assist in gathering labelled training data.  Writing metadata to file, generating annotations, image manipulation

# Installation
git clone https://ccweaver30@bitbucket.org/blacksquirrelsintel/data_gathering_tools.git


# Usage
## Using the drag and drop alignment for M-matrix generation
change the 'data_dir' inside drag_display.py.  This should point to a directory filled with XML files.

python drag_display.py
-First drag the circles on top of the image points you would like to warp based on
-Hit 'Swtich Mode'
-Drag the circles in order to warp and align the rink and the map
-When finished, hit 'Next & Save'.  This generates a .json file in /out directory with the corresponding M-matrix for warping