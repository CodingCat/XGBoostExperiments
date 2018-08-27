package me.codingcat.xgboost

import ml.dmlc.xgboost4j.scala.spark.XGBoostEstimator

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.{DoubleType, StringType, StructField, StructType}

object IrisPipeline {

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().getOrCreate()
    val schema = new StructType(Array(
      StructField("sepal length", DoubleType, true),
      StructField("sepal width", DoubleType, true),
      StructField("petal length", DoubleType, true),
      StructField("petal width", DoubleType, true),
      StructField("class", StringType, true)))
    val rawInput = spark.read.schema(schema).csv(args(0))

    // transform class to index to make xgboost happy
    val stringIndexer = new StringIndexer()
      .setInputCol("class")
      .setOutputCol("classIndex")
      // .fit(rawInput)
    // val labelTransformed = stringIndexer.transform(rawInput).drop("class")
    // compose all feature columns as vector
    val vectorAssembler = new VectorAssembler().
      setInputCols(Array("sepal length", "sepal width", "petal length", "petal width")).
      setOutputCol("features")
    // val xgbInput = vectorAssembler.transform(labelTransformed).select("features",
      // "classIndex")
    val xgbParam = Map("eta" -> 0.1f,
      "max_depth" -> 2,
      "objective" -> "multi:softprob",
      "num_class" -> 3,
      "num_round" -> 100,
      "num_workers" -> 2)

    val xgbEstimator = new XGBoostEstimator(xgbParam)
    xgbEstimator.setFeaturesCol("features")
    xgbEstimator.setLabelCol("classIndex")

    val pipeline = new Pipeline()
    pipeline.setStages(Array(stringIndexer, vectorAssembler, xgbEstimator))
    val pipelineModel = pipeline.fit(rawInput)
    pipelineModel.transform(rawInput).show()
    pipelineModel.write.overwrite().save(args(1))
  }
}
