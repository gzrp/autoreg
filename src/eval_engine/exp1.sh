#!/bin/bash


echo "===================================="
echo "Dataset: diabetic"

files=("reg_selection_new.py")

echo "===================================="
echo "The following scripts will be executed in order:"
for f in "${files[@]}"; do
    echo " - $f"
done

echo "===================================="
echo "grace_period=1"

for f in "${files[@]}"; do
    echo "------------------------------------"
    echo "Running: $f"

    python "$f" --num_samples=2000 --dataset=diabetic --batch_size=64 --grace_period=2 --max_epochs=4
    status=$?

    if [ $status -ne 0 ]; then
        echo "$f failed with exit code $status. Skipping to next script."
    else
        echo "$f completed successfully."
    fi

    echo "Waiting 5 seconds before next script..."
    sleep 5
done

echo ""
echo "All scripts have finished running."
echo "------------------------------------"