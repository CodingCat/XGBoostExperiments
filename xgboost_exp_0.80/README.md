# Migrate 0.6 Pipeline Model to 0.8

command line 

1. build the migration tool

```bash
git clone git@github.com:CodingCat/XGBoostExperiments.git

cd XGBoostExperiments

mvn package

```

2. run migration program


```bash

# ensure that you have spark-2.3.0+ Spark version
# you can also specify parameters to claim more resources if `.fit()` takes resources

spark-submit --class me.codingcat.xgboost.PipelineLoader xgboost_exp_0.80/target/xgboost_exp_0.80-0.1-SNAPSHOT-jar-with-dependencies.jar \
rootPathOfPipelineModel outputPath datasetPath

```

* <b>rootPathOfPipelineModel</b>: the location of model directory (containing `metadata` and `stages` sub-folders)

* <b>outputPath</b>: the location to save 0.8 version of pipeline model 

* <b>datasetPath</b>: the location of the dataset to be fit by PipelineModel
