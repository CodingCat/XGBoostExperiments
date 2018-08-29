package ml.dmlc.xgboost4j.scala.spark

import ml.dmlc.xgboost4j.scala.Booster

class XGBoostRegressionModelBridge(uid: String, booster: Booster) {


  val xgbRegressionModel = new XGBoostRegressionModel(uid, booster)
}
