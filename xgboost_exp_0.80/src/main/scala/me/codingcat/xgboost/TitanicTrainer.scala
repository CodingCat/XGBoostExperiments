package me.codingcat.xgboost

import scala.collection.mutable

import ml.dmlc.xgboost4j.scala.spark.XGBoostClassifier

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.sql.SparkSession

object TitanicTrainer {

  case class Passenger(
    id: Long,
    survived: Double,
    pClass: Long,
    name: String,
    sex: String,
    age: Double,
    sibSp: Long,
    parCh: Long,
    ticket: String,
    fare: Double,
    cabin: String,
    embarked: String)

  def main(args: Array[String]): Unit = {
    val path = args(0)
    val spark = SparkSession.builder().getOrCreate()
    val inputDS = spark.read.format("csv").option("inferSchema", true).load(path)
    val inputDF = inputDS.toDF("id", "survived", "pClass", "name", "sex", "age",
      "sibSp", "parCh", "ticket", "fare", "cabin", "embarked")
    val stringIndexer1 = new StringIndexer().setHandleInvalid("keep").
      setInputCol("sex").setOutputCol("sexIndex")
    val stringIndexer2= new StringIndexer().setHandleInvalid("keep").
      setInputCol("ticket").setOutputCol("ticketIndex")
    val stringIndexer3= new StringIndexer().setHandleInvalid("keep").
      setInputCol("cabin").setOutputCol("cabinIndex")
    val stringIndexer4= new StringIndexer().setHandleInvalid("keep").
      setInputCol("embarked").setOutputCol("embarkedIndex")
    // todo: one hot
    val vectorAssembler = new VectorAssembler().setInputCols(
      Array("pClass", "sexIndex", "age", "sibSp", "parCh", "ticketIndex", "fare", "cabinIndex",
        "embarkedIndex")).setOutputCol("features")
    val pipeline = new Pipeline().setStages(Array(stringIndexer1,
      stringIndexer2, stringIndexer3, stringIndexer4))
    val transformedDF = pipeline.fit(inputDF).transform(inputDF)
    val assembledDF = vectorAssembler.transform(transformedDF.na.drop)
    val Array(train, test) = assembledDF.randomSplit(Array(0.8, 0.2))
    val classifier = new XGBoostClassifier().setFeaturesCol("features").setLabelCol("survived")
    classifier.setTreeMethod("hist")
    classifier.setNumWorkers(Runtime.getRuntime.availableProcessors())
    classifier.setNumRound(100)
    classifier.setObjective("binary:logistic")
    val model = classifier.fit(train)
    val result = model.transform(test)
    import org.apache.spark.sql.functions._
    val resultFlatten = result.withColumn("raw", udf {
      rawPrediction: Vector =>
        rawPrediction.toArray(0)
    }.apply(col("rawPrediction")))
    resultFlatten.show()
    val evaluator = new BinaryClassificationEvaluator()
    evaluator.setLabelCol("survived").setRawPredictionCol("probability")
    println(evaluator.evaluate(resultFlatten))
  }
}
