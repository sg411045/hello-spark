import org.apache.spark.SparkContext
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.SQLContext
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator


object IrisClassification {
  def main(args: Array[String]) {
      val spark = SparkSession.builder.appName("Iris Classification").config("spark.master", "local").getOrCreate()
      var data = spark.read.option("header", false).option("inferschema", true)
        .csv("./iris.data")
        .toDF("sepal_length","sepal_width", "petal_length", "petal_width", "label") 
      val assembler = new VectorAssembler().setInputCols(Array("sepal_length","sepal_width", "petal_length", "petal_width")).setOutputCol("features")
      data = assembler.transform(data)
      val df = data.drop("sepal_length","sepal_width", "petal_length", "petal_width")
      val indexer = new StringIndexer().setInputCol("label").setOutputCol("labelIndex")
      val df_1 = indexer.fit(df).transform(df)
      val splits = df_1.randomSplit(Array(0.7, 0.3))

      val dt = new DecisionTreeClassifier().setLabelCol("labelIndex").setFeaturesCol("features")
      val model = dt.fit(splits(0))

      val predictions = model.transform(splits(1))
      val evaluator = new MulticlassClassificationEvaluator().setLabelCol("labelIndex").setPredictionCol("prediction").setMetricName("accuracy")
      val accuracy = evaluator.evaluate(predictions)

      println("decision tree classifier accuracy ", accuracy)

      // val rf = new RandomForestClassifier().setLabelCol("labelIndex").setFeaturesCol("features").setNumTrees(10)
      spark.stop()

  } 

}
