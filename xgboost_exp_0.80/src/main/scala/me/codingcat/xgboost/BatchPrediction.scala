package me.codingcat.xgboost

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

import ml.dmlc.xgboost4j.LabeledPoint
import ml.dmlc.xgboost4j.java.Rabit
import ml.dmlc.xgboost4j.scala.{Booster, DMatrix}
import ml.dmlc.xgboost4j.scala.spark.XGBoostClassifier

import org.apache.spark.TaskContext
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vector => SparkVector}
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.sql.types.{UserDefinedType, _}

object BatchPrediction {

  private implicit class MLVectorToXGBLabeledPoint(val v: SparkVector) extends AnyVal {
    /**
     * Converts a [[SparkVector]] to a data point with a dummy label.
     *
     * This is needed for constructing a [[ml.dmlc.xgboost4j.scala.DMatrix]]
     * for prediction.
     */
    def asXGB: LabeledPoint = v match {
      case v: DenseVector =>
        LabeledPoint(0.0f, null, v.values.map(_.toFloat))
      case v: SparseVector =>
        LabeledPoint(0.0f, v.indices, v.values.map(_.toFloat))
    }
  }
  def create(model: Booster, arrayVecDataset: DataFrame, spark: SparkSession): DataFrame = {
    spark.createDataFrame(arrayVecDataset.rdd.zipPartitions(
      arrayVecDataset.rdd.mapPartitions {
        rows =>
          import scala.collection.JavaConverters._
          val rabitEnv = Array("DMLC_TASK_ID" -> TaskContext.getPartitionId().toString).toMap
          Rabit.init(rabitEnv.asJava)
          val arrayed_features = rows.map(_.getAs[mutable.WrappedArray[SparkVector]]("array_features"))
          val indices = new ArrayBuffer[Int]
          var currentIndex = 0
          val flattenedVectors = arrayed_features.flatMap {
            vectorArray: mutable.WrappedArray[SparkVector] =>
              vectorArray.foreach { v =>
                indices += currentIndex
              }
              currentIndex += 1
              vectorArray
          }
          val predictResults = model.predict(new DMatrix(flattenedVectors.map(_.asXGB)))
          val finalArrayedResults = predictResults.zip(indices).groupBy(_._2).map(_._2.map(_._1))
          finalArrayedResults.iterator
      }
    )((rows, finalArrayedResults) =>
      rows.zip(finalArrayedResults).map {
        case (row, results) =>
          Row.merge(row, Row(results))
      }
    ), arrayVecDataset.schema.add(StructField("arrayed_output", ArrayType(ArrayType(FloatType)))))
  }


  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().getOrCreate()
    import spark.implicits._
    val schema = new StructType(Array(
      StructField("sepal length", DoubleType, true),
      StructField("sepal width", DoubleType, true),
      StructField("petal length", DoubleType, true),
      StructField("petal width", DoubleType, true),
      StructField("class", StringType, true)))
    val Array(trainingSet, testSet) = spark.read.schema(schema).csv(args(0)).randomSplit(Array(0.8, 0.2))

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

    val pipeline = new Pipeline().setStages(Array(stringIndexer, vectorAssembler))
    val pipelineModel = pipeline.fit(trainingSet)

    val transformedTrain = pipelineModel.transform(trainingSet)
    val model = xgboostClassifier.fit(transformedTrain)

    // build a DataFrame with Array[Vector] as column type
    val transformedTest = pipelineModel.transform(testSet)
    val arrayVecDataset = transformedTest.withColumn("array_features", udf {
      v: SparkVector =>
        Array(v)
    }.apply(col("features")))
    arrayVecDataset.show()
    // do prediction
    create(model.nativeBooster, arrayVecDataset, spark).show()

  }
}
