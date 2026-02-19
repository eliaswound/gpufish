gpu fish: 
A gpu compatible smFISH detection platform for processing large tissue and sections
Inspired by bigfish/fishquant 

Improvements: 
Compact core function library make it lighter and easier to maintain 
Improved heuristic functions to find thresholds and filter out spots in background 
gpu compatibility for large scale image processing using gpu. 
batch progress in folder 

Requiremnt packages 
torch 
cucim 
cupy 
tqdm 
cellpose