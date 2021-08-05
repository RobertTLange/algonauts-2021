conda create -n algonauts python=3.8
source activate algonauts
curl https://raw.githubusercontent.com/automl/auto-sklearn/master/requirements.txt | xargs -n 1 -L 1 pip3 install
pip3 install auto-sklearn
pip install -r requirements.txt

# Install mle-logging and mle-toolbox
# pip install mle-toolbox
