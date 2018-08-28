package me.codingcat.xgboost

import org.apache.spark.ml.PipelineModel

object PipelineLoader {
  def main(args: Array[String]): Unit = {
    val pipelineModel = PipelineModel.load(args(0))
    pipelineModel.stages.foreach(stage => println(stage.uid))
  }
}
