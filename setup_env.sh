
set -e

# conda create -n robobrain2 python=3.10
# conda activate robobrain2
# clone repo.


git clone https://github.com/FlagOpen/RoboBrain2.0.git
cd RoboBrain2.0
pip install -r requirements.txt

# install sam2
git clone https://github.com/facebookresearch/sam2.git && cd sam2
pip install -e .
pip install -e ".[notebooks]"
cd checkpoints && \
./download_ckpts.sh && \

cd ..
mv sam2/checkpoints -t .
