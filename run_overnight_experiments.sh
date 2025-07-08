#!/bin/bash
export PYTHONPATH=$PYTHONPATH:.
export TOKENIZERS_PARALLELISM=false
echo "======================================================"
echo "      LAUNCHING OVERNIGHT RESEARCH"
echo "======================================================"

# --- Path A ---
(python src/trainer.py --config path_a1_config.yaml && python analyze_models.py --exp_id "path_a1_gated_hybrid_gate0.05" --gpu 0) &
(python src/trainer.py --config path_a2_config.yaml && python analyze_models.py --exp_id "path_a2_gated_hybrid_gate0.2" --gpu 1) &
(python src/trainer.py --config path_a3_config.yaml && python analyze_models.py --exp_id "path_a3_gated_hybrid_mask0.9" --gpu 2) &
(python src/trainer.py --config path_a4_config.yaml && python analyze_models.py --exp_id "path_a4_gated_absorbing" --gpu 3) &

# --- Path B (Illustrative - requires new trainer script) ---
# (python src/trainer_twostream.py --config path_b_twostream_config.yaml && python analyze_models.py --exp_id "path_b_twostream" --gpu 4) &

# --- Path C (Illustrative - requires new trainer script) ---
# (python src/trainer_progressive.py --config path_c_progressive_config.yaml && python analyze_models.py --exp_id "path_c_progressive" --gpu 6) &


echo "All experiments launched. Waiting for completion..."
wait
echo "All experiments have completed."
