#!/bin/bash


echo "===================================="
echo "Dataset: devnagari"

files=("exp2_asha.py" "exp2_bohb.py" "exp2_hyperband.py")
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

    python "$f" --num_samples=2000 --dataset=frappe --batch_size=128
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