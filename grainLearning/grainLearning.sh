#!/bin/sh
#File information and yade version 
pythonVersion="python3"
pythonFile="grainLearning.py"

# Initialisation
command="$pythonVersion input_declaration.py"
echo "Read input_file"
eval $command
command="$pythonVersion $pythonFile -1"
echo "Perform initialisation step ..."
eval $command



# Loop over grain learning steps 
iterNO="0"
runSimulation=true

while $runSimulation ;do 
 # perform 1 grain learning step and start simulations if not done 
 command="$pythonVersion $pythonFile $iterNO"
 echo "Perform  step $iterNO"
 eval $command
 
 # increment iteration number 
 iterNO="$(($iterNO + 1))"
 # check if GL is done (file finishedGL.txt is generated via python)
 if [ -f "finishedGL.txt" ]
 then
    iterNO="$(($iterNO - 1))"
    runSimulation=false
 fi
done
echo "Grain learning is done"
