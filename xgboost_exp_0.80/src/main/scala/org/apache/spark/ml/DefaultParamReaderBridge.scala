package org.apache.spark.ml

import org.apache.spark.SparkContext
import org.apache.spark.ml.util.DefaultParamsReader

object DefaultParamReaderBridge {
  def loadParamsInstance[T](path: String, sc: SparkContext): T = {
    DefaultParamsReader.loadParamsInstance[T](path, sc)
  }
}
