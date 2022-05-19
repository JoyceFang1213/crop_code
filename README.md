# README

## Environment
```shell
$ pip install -r requirements.txt
```

## Training
There are nine training files in the folder, each file provide one training model. To set up the training files, users should change the training and testing folder listed in `ImageFolder`.

```shell
# Running the Python script
# e.x. python main_swin_base_224.py
$ python <file>
```

## Prediction
To get the result, run:
```shell
$ python predict.py
```
> Note that the users should change the **weight path** and **save place** of csv in `predict.py`.

## Merging
Run the `merge.ipynb` and fill up the the csv file's path. The generated file is called `res.csv`.