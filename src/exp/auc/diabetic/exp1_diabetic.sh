#!/bin/bash


echo "===================================="
echo "Dataset: diabetic"

#files=("exp2_asha.py" "exp2_bohb.py" "exp2_hyperband.py" "exp1_2phase.py")
files=("../exp2_asha.py")
echo "===================================="
echo "The following scripts will be executed in order:"
for f in "${files[@]}"; do
    echo " - $f"
done
echo "===================================="
echo ""

for f in "${files[@]}"; do
    echo "------------------------------------"
    echo "Running: $f"

    python "$f" --dataset=diabetic --batch_size=128 --seed=42 --num_samples=10000 --max_epochs=16 --device_ids=2,3 --gpu_ids=2,3,2,3
    status=$?

    if [ $status -ne 0 ]; then
        echo "$f failed with exit code $status. Skipping to next script."
    else
        echo "$f completed successfully."
    fi

    echo "Waiting 3 seconds before next script..."
    sleep 3
done

echo ""
echo "All scripts have finished running."
echo "------------------------------------"