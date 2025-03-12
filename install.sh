python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
python3 -m pip install numpy pandas matplotlib scikit-learn torchinfo torchmetrics tabulate
cd ./data/transformers
python3 -m pip install .
cd ../../

python3 -m pip install evaluate datasets accelerate
