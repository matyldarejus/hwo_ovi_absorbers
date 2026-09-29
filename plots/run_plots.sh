#!/bin/bash
 
MODEL="m100n1024"
WIND="s50"
SNAP="151"
 
SCRIPT_DIR="/home/matylda/sh/plots/reworked"
 
echo "Running all plot scripts for ${MODEL} ${WIND} ${SNAP}..."
 
run_script() {
    echo "----------------------------------------"
    echo "Running: $1"
    python "$1" "${@:2}"
    if [ $? -eq 0 ]; then
        echo "Done: $1"
    else
        echo "ERROR: $1 failed — continuing..."
    fi
}
 
run_script "${SCRIPT_DIR}/plot_galaxy_sample.py"      "$MODEL" "$WIND" "$SNAP"
run_script "${SCRIPT_DIR}/plot_ew_profile.py"         "$MODEL" "$WIND" "$SNAP" 8 False
run_script "${SCRIPT_DIR}/plot_cddf.py"               "$MODEL" "$WIND" "$SNAP"
run_script "${SCRIPT_DIR}/plot_cddf_mass.py"          "$MODEL" "$WIND" "$SNAP"
run_script "${SCRIPT_DIR}/plot_cddf_rho.py"           "$MODEL" "$WIND" "$SNAP"
run_script "${SCRIPT_DIR}/plot_cddf_az.py"            "$MODEL" "$WIND" "$SNAP"
run_script "${SCRIPT_DIR}/plot_b_distributions.py"    "$MODEL" "$WIND" "$SNAP"
run_script "${SCRIPT_DIR}/plot_NT.py"                 "$MODEL" "$WIND" "$SNAP"
run_script "${SCRIPT_DIR}/plot_col_prof.py"           "$MODEL" "$WIND" "$SNAP"
run_script "${SCRIPT_DIR}/plot_Z.py"                  "$MODEL" "$WIND" "$SNAP"
run_script "${SCRIPT_DIR}/plot_Z_bins.py"             "$MODEL" "$WIND" "$SNAP"
run_script "${SCRIPT_DIR}/plot_Z_medians.py"          "$MODEL" "$WIND" "$SNAP"
 
echo "----------------------------------------"
echo "All scripts completed."