mamba deactivate
mamba env remove -n llmtrain -y

conda create -n llmtrain python=3.12 -y
mamba activate llmtrain
conda activate llmtrain

pip install scikit-learn
conda install transformers numpy -c conda-forge -y
conda install pytorch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 pytorch-cuda=12.4 -c pytorch -c nvidia -y
conda install pytorch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 pytorch-cuda=12.4 -y