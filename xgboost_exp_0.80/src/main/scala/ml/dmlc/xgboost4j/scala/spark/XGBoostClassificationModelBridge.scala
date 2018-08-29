package ml.dmlc.xgboost4j.scala.spark

import ml.dmlc.xgboost4j.scala.Booster

class XGBoostClassificationModelBridge(
    uid: String,
    numClasses: Int,
    _booster: Booster) {
  val xgbClassificationModel = new XGBoostClassificationModel(uid, numClasses, _booster)
}
