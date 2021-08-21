conda create -n algonauts python=3.8
source activate algonauts
# Install mle-toolbox & other dependencies
pip install -r requirements.txt
# Install auto-sklearn
curl https://raw.githubusercontent.com/automl/auto-sklearn/master/requirements.txt | xargs -n 1 -L 1 pip3 install
pip install auto-sklearn
