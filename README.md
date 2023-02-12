# FMDTA
## Source codes:
create_data.py: create data in pytorch format<br>
utils.py: include TestbedDataset used by create_data.py to create data, and performance measures.<br>
training.py: train the FMDTA model.<br>
ginconv.py： model file.<br>

## Dependencies:
python==3.7<br>
pytorch==1.5<br>
numpy == 1.17.4<br>

## Dataset：
Our drug molecular graph data were obtained from GraphDTA（https://github.com/thinng/GraphDTA）<br>
Our target structure graph data were obtained from DGraphDTA（https://github.com/595693085/DGraphDTA）<br>
python create_data.py
This returns  davis_train.csv, and davis_test.csv, kiba_train.csv, kiba_test.csv.<br>

## Train Model
Train a prediction model<br>
python training.py 0 0 0
