package me.codingcat.xgboost

import ml.dmlc.xgboost4j.scala.XGBoost
import ml.dmlc.xgboost4j.scala.spark.{XGBoostClassificationModel, XGBoostClassificationModelBridge, XGBoostRegressionModel, XGBoostRegressionModelBridge}
import org.apache.hadoop.fs.{FSDataInputStream, FileStatus, FileSystem, Path}

import org.apache.spark.ml.util.Identifiable
import org.apache.spark.ml.{DefaultParamReaderBridge, Pipeline, PipelineModel, PipelineStage}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.types.{DoubleType, StringType, StructField, StructType}

object PipelineLoader {

  private def loadGeneralModelParams(inputStream: FSDataInputStream): (String, String, String) = {
    val featureCol = inputStream.readUTF()
    val labelCol = inputStream.readUTF()
    val predictionCol = inputStream.readUTF()
    (featureCol, labelCol, predictionCol)
  }

  private def loadRegressionModel(dataFile: String, uid: String, fs: FileSystem):
      XGBoostRegressionModel = {
    val dataInStream = fs.open(new Path(dataFile))
    // skip modelType
    dataInStream.readUTF()
    val (featureCol, _, predictionCol) = loadGeneralModelParams(dataInStream)
    val regressionModel = new XGBoostRegressionModelBridge(Identifiable.randomUID("xgbr"),
      XGBoost.loadModel(dataInStream)).
      xgbRegressionModel
    regressionModel.setFeaturesCol(featureCol)
    regressionModel.setPredictionCol(predictionCol)
    regressionModel
  }

  private def loadClassificationModel(
      dataFile: String, uid: String, fs: FileSystem): XGBoostClassificationModel = {
    val dataInStream = fs.open(new Path(dataFile))
    // skip modelType
    dataInStream.readUTF()
    val (featureCol, _, predictionCol) = loadGeneralModelParams(dataInStream)
    val rawPredictionCol = dataInStream.readUTF()
    val numClasses = dataInStream.readInt()
    val thresholdLength = dataInStream.readInt()
    var thresholds: Array[Double] = null
    if (thresholdLength != -1) {
      thresholds = new Array[Double](thresholdLength)
      for (i <- 0 until thresholdLength) {
        thresholds(i) = dataInStream.readDouble()
      }
    }
    val xgBoostModel = new XGBoostClassificationModelBridge(Identifiable.randomUID("xgbc"),
      numClasses, XGBoost.loadModel(dataInStream)).xgbClassificationModel
    xgBoostModel.setRawPredictionCol(rawPredictionCol)
    xgBoostModel.setFeaturesCol(featureCol)
    xgBoostModel.setPredictionCol(predictionCol)
    if (thresholds != null) {
      xgBoostModel.setThresholds(thresholds)
    }
    xgBoostModel
  }

  private def loadTrainingSet(inputFile: String, spark: SparkSession): DataFrame = {
    val schema = new StructType(Array(
      StructField("sepal length", DoubleType, true),
      StructField("sepal width", DoubleType, true),
      StructField("petal length", DoubleType, true),
      StructField("petal width", DoubleType, true),
      StructField("class", StringType, true)))
    spark.read.schema(schema).csv(inputFile)
  }


  def main(args: Array[String]): Unit = {
    val sparkSession = SparkSession.builder().getOrCreate()
    val rootPathOfPipelineModel = args(0)
    val outputPath = args(1)
    val trainingSet = loadTrainingSet(args(2), sparkSession)
    val stagesPath = new Path(s"$rootPathOfPipelineModel/stages")
    val fs = stagesPath.getFileSystem(sparkSession.sparkContext.hadoopConfiguration)
    val allStageDirs = fs.listStatus(stagesPath)
    val newStages = new Array[PipelineStage](allStageDirs.length)
    allStageDirs.foreach { stageDir: FileStatus =>
      val stageDirPath = stageDir.getPath
      val Array(stageId, stageName, uidHashCode) = stageDirPath.getName.split("_")
      val uid = s"${stageName}_$uidHashCode"
      val stage = {
        if (!uid.contains("XGBoost")) {
          DefaultParamReaderBridge.loadParamsInstance[PipelineStage](
            stageDirPath.toString, sparkSession.sparkContext)
        } else if (uid.contains("Classification")) {
          // val dataFile = s"${stageDirPath.toString}/data.json"
          val dataFile = s"${stageDirPath.toString}/data"
          loadClassificationModel(dataFile, uid, fs)
        } else if (uid.contains("Regression")) {
          val dataFile = s"${stageDirPath.toString}/data"
          loadRegressionModel(dataFile, uid, fs)
        } else {
          throw new Exception("Unrecognizable directory")
        }
      }
      newStages(stageId.toInt) = stage
    }
    val pipeline = new Pipeline()
    pipeline.setStages(newStages)
    val pipelineModel = pipeline.fit(trainingSet)
    pipelineModel.write.overwrite().save(outputPath)
    // pipeline.getStages.foreach(stage => println(stage.uid))
  }
}
