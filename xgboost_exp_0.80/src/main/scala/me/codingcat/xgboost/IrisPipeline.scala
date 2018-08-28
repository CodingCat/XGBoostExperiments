package me.codingcat.xgboost

import ml.dmlc.xgboost4j.scala.spark.XGBoostClassifier

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

    val vectorAssembler = new VectorAssembler().
      setInputCols(Array("sepal length", "sepal width", "petal length", "petal width")).
      setOutputCol("features")

    val xgbParam = Map("eta" -> 0.1f,
      "max_depth" -> 2,
      "objective" -> "multi:softprob",
      "num_class" -> 3,
      "num_round" -> 100,
      "num_workers" -> 2)
    val xgboostClassifier = new XGBoostClassifier(xgbParam)
    xgboostClassifier.setFeaturesCol("features")
    xgboostClassifier.setLabelCol("classIndex")

    val pipeline = new Pipeline().setStages(Array(stringIndexer, vectorAssembler, xgboostClassifier))
    val pipelineModel = pipeline.fit(rawInput)
    pipelineModel.stages.foreach(stage => println(stage.uid))
    pipelineModel.write.overwrite().save(args(1))
  }
}
