#!/bin/bash
# Quick launcher for all TTT model training

echo "=========================================="
echo "TTT Model Training on C4 Dataset"
echo "=========================================="
echo ""
echo "Available models:"
echo "  1) TTT-125M  (4.8K steps,  ~2.4B tokens)"
echo "  2) TTT-350M  (13.5K steps, ~6.75B tokens)"
echo "  3) TTT-760M  (29K steps,   ~14.5B tokens)"
echo "  4) TTT-1B    (50K steps,   ~25B tokens)"
echo ""
echo "  0) Exit"
echo ""
read -p "Select model to train: " choice

case $choice in
    1)
        echo "Launching TTT-125M training..."
        ./train_125m_c4.sh
        ;;
    2)
        echo "Launching TTT-350M training..."
        ./train_350m_c4.sh
        ;;
    3)
        echo "Launching TTT-760M training..."
        ./train_760m_c4.sh
        ;;
    4)
        echo "Launching TTT-1B training..."
        ./train_1b_c4.sh
        ;;
    0)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo "Invalid choice. Exiting..."
        exit 1
        ;;
esac
