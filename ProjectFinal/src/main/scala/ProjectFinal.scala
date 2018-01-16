
import org.apache.log4j
import org.apache.log4j.{LogManager, Logger}
import org.apache.spark.sql.functions._
import org.apache.spark.sql._
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorAssembler}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.classification.LogisticRegression
import scala.collection.mutable.ListBuffer
import org.apache.spark.ml.classification.LinearSVC
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.classification.GBTClassifier
import org.apache.spark.ml.classification.DecisionTreeClassifier

object Project {
  def main(args: Array[String]) {

    val log = LogManager.getRootLogger
    log.info("Start")
    if (args.length<1){
      log.error("Argument missing")
      return
    }
    val inputmifemData = args(0)
    val outputCorrelation = args(1)
    val outputRandomModel=args(2)
    val outputRandomConfusion=args(3)
    val outputLogisticModel=args(4)
    val outputLogisticConfusion=args(5)
    val outputLinearSVMModel=args(6)
    val outputLinearSVMConfusion=args(7)
    val outputNaiveBayesModel=args(8)
    val outputNaiveBayesConfusion=args(9)
    val outputGBTModel=args(10)
    val outputGBTConfusion=args(11)
    val outputDTModel=args(12)
    val outputDTConfusion=args(13)
    val outputAccuracies = args(14)

    val conf = new SparkConf().setAppName("SHProject")
    val sc = new SparkContext(conf)
    val spark = SparkSession.builder().appName("SHProject").getOrCreate()

    val mifemData = spark.read.format("csv").option("header",true).option("inferschema",true).option("sep",",").load(inputmifemData)
    val mifemCleanData = mifemData.na.drop
    mifemCleanData.printSchema()

    val mifemDataset = mifemCleanData.select("outcome","age","yronset","premi","smstat","diabetes","highbp","hichol","angina","stroke")

    val categoryColNames = List("premi","smstat","diabetes","highbp","hichol","angina","stroke")
    val stringIndexersFeatures = categoryColNames.map { colName =>
      new StringIndexer()
        .setInputCol(colName)
        .setOutputCol(colName + "Indexed")
        .fit(mifemDataset)
    }

    val outcomeIndexer = new StringIndexer()
      .setInputCol("outcome")
      .setOutputCol("outcomeIndexed")
      .fit(mifemDataset)

    var catColNameIndexed = new ListBuffer[String]()

    catColNameIndexed = categoryColNames.map(_ + "Indexed").to[ListBuffer]
    catColNameIndexed += "age"
    catColNameIndexed += "yronset"

    val assembler = new VectorAssembler()
      .setInputCols(Array(catColNameIndexed: _*))
      .setOutputCol("Features")

    val pipelineMifem = new Pipeline().setStages(
      Array(stringIndexersFeatures: _*)++ Array(outcomeIndexer,assembler))

    val indexedData = pipelineMifem.fit(mifemDataset).transform(mifemDataset)

    val outcomeFeatureData = indexedData.select("outcome","outcomeIndexed","Features")

    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(outcomeIndexer.labels)

    val corr1 = indexedData.stat.corr("outcomeIndexed","age")
    var correlation = "age : " + corr1 + "\n"
    val corr2 = indexedData.stat.corr("outcomeIndexed","yronset")
    correlation += "yronset : " + corr2 + "\n"
    val corr3 = indexedData.stat.corr("outcomeIndexed","premiIndexed")
    correlation += "premiIndexed : " + corr3 + "\n"
    val corr4 = indexedData.stat.corr("outcomeIndexed","smstatIndexed")
    correlation += "smstatIndexed : " + corr4 + "\n"
    val corr5 = indexedData.stat.corr("outcomeIndexed","diabetesIndexed")
    correlation += "diabetesIndexed : " + corr5 + "\n"
    val corr6 = indexedData.stat.corr("outcomeIndexed","highbpIndexed")
    correlation += "highbpIndexed : " + corr6 + "\n"
    val corr7 = indexedData.stat.corr("outcomeIndexed","hicholIndexed")
    correlation += "hicholIndexed : " + corr7 + "\n"
    val corr8 = indexedData.stat.corr("outcomeIndexed","anginaIndexed")
    correlation += "anginaIndexe : " + corr8 + "\n"
    val corr9 = indexedData.stat.corr("outcomeIndexed","strokeIndexed")
    correlation += "strokeIndexed : " + corr9 + "\n"

    val printing = sc.parallelize(Seq(correlation))
    printing.saveAsTextFile(outputCorrelation)

    val randomForestModel = new RandomForestClassifier()
      .setLabelCol("outcomeIndexed")
      .setFeaturesCol("Features")

    val logisticRegressionModel = new LogisticRegression()
      .setLabelCol("outcomeIndexed")
      .setFeaturesCol("Features")

    val linearSVMModel = new LinearSVC()
      .setMaxIter(20)
      .setRegParam(0.1)
      .setLabelCol("outcomeIndexed")
      .setFeaturesCol("Features")

    val naiveBayesModel = new NaiveBayes()
      .setLabelCol("outcomeIndexed")
      .setFeaturesCol("Features")

    val gbtModel = new GBTClassifier()
      .setMaxIter(10)
      .setLabelCol("outcomeIndexed")
      .setFeaturesCol("Features")

    val decisionTreeModel = new DecisionTreeClassifier()
      .setLabelCol("outcomeIndexed")
      .setFeaturesCol("Features")

    val pipelineRandomForest = new Pipeline().setStages(Array(randomForestModel,labelConverter))
    val pipelineLogisticRegression = new Pipeline().setStages(Array(logisticRegressionModel,labelConverter))
    val pipelineLinearSVM = new Pipeline().setStages(Array(linearSVMModel,labelConverter))
    val pipelineNaiveBayes = new Pipeline().setStages(Array(naiveBayesModel,labelConverter))
    val pipelineGBT = new Pipeline().setStages(Array(gbtModel,labelConverter))
    val pipelineDecisionTree = new Pipeline().setStages(Array(decisionTreeModel,labelConverter))

    val Array(trainingData, testData) = outcomeFeatureData.randomSplit(Array(0.7, 0.3))

    val randomForest = pipelineRandomForest.fit(trainingData)
    randomForest.save(outputRandomModel)
    val predictionsRandomForest = randomForest.transform(testData)

    val logisticRegression = pipelineLogisticRegression.fit(trainingData)
    logisticRegression.save(outputLogisticModel)
    val predictionsLogistic = logisticRegression.transform(testData)

    val linearSVM = pipelineLinearSVM.fit(trainingData)
    linearSVM.save(outputLinearSVMModel)
    val predictionsLinearSVM = linearSVM.transform(testData)

    val naiveBayes = pipelineNaiveBayes.fit(trainingData)
    naiveBayes.save(outputNaiveBayesModel)
    val predictionsNaiveBayes = naiveBayes.transform(testData)

    val gbt = pipelineGBT.fit(trainingData)
    gbt.save(outputGBTModel)
    val predictionsGBT = gbt.transform(testData)

    val decisionTree = pipelineDecisionTree.fit(trainingData)
    decisionTree.save(outputDTModel)
    val predictionsDecisionTree = decisionTree.transform(testData)

    val randomForestCorrect = correctPredictions(predictionsRandomForest)
    val randomForestWrong = wrongPredictions(predictionsRandomForest)

    val (randomAccuracy,matrixRandom) = accuracy(randomForestCorrect,randomForestWrong,testData)
    var accuracyString = "RandomForest : " + randomAccuracy + "\n"
    val confRandom = sc.parallelize(Seq(matrixRandom))
    confRandom.saveAsTextFile(outputRandomConfusion)

    val logisticCorrect = correctPredictions(predictionsLogistic)
    val logisticWrong = wrongPredictions(predictionsLogistic)


    val (logisticAccuracy,matrixLogistic)= accuracy(logisticCorrect,logisticWrong,testData)
    accuracyString += "logisticRegression : " + logisticAccuracy + "\n"
    val confMatrixLogistic = sc.parallelize(Seq(matrixLogistic))
    confMatrixLogistic.saveAsTextFile(outputLogisticConfusion)

    val linearCorrect = correctPredictions(predictionsLinearSVM)
    val linearWrong = wrongPredictions(predictionsLinearSVM)

    val (linearSVMAccuracy,matrixLinearSVM) = accuracy(linearCorrect,linearWrong,testData)
    accuracyString += "linearSVM : " + linearSVMAccuracy + "\n"
    val confMatrixLinearSVM = sc.parallelize(Seq(matrixLinearSVM))
    confMatrixLinearSVM.saveAsTextFile(outputLinearSVMConfusion)

    val naiveBayesCorrect = correctPredictions(predictionsNaiveBayes)
    val naiveBayesWrong = wrongPredictions(predictionsNaiveBayes)

    val (naiveBayesAccuracy,matrixNaiveBayes) = accuracy(naiveBayesCorrect,naiveBayesWrong,testData)
    accuracyString += "NaiveBayes: " + naiveBayesAccuracy + "\n"
    val confMatrixNaiveBayes = sc.parallelize(Seq(matrixNaiveBayes))
    confMatrixNaiveBayes.saveAsTextFile(outputNaiveBayesConfusion)

    val gbtCorrect = correctPredictions(predictionsGBT)
    val gbtWrong = wrongPredictions(predictionsGBT)

    val (gbtAccuracy,matrixGBT)= accuracy(gbtCorrect,gbtWrong,testData)
    accuracyString += "GradientBoostingTree : " + gbtAccuracy + "\n"
    val confMatrixGBT = sc.parallelize(Seq(matrixGBT))
    confMatrixGBT.saveAsTextFile(outputGBTConfusion)

    val decisionTreeCorrect = correctPredictions(predictionsDecisionTree)
    val decisionTreeWrong = wrongPredictions(predictionsDecisionTree)

    val (decisionTreeAccuracy,matrixDecisionTree) = accuracy(decisionTreeCorrect,decisionTreeWrong,testData)
    accuracyString += "DecisionTree : " + decisionTreeAccuracy + "\n"
    val confMatrixDT = sc.parallelize(Seq(matrixDecisionTree))
    confMatrixDT.saveAsTextFile(outputDTConfusion)

    val accuracyCompare = sc.parallelize(Seq(accuracyString))
    accuracyCompare.saveAsTextFile(outputAccuracies)

    sc.stop()
  }
  def correctPredictions(predictions: org.apache.spark.sql.Dataset[org.apache.spark.sql.Row] ): DataFrame = {
    val correctPredictions = predictions.where(expr("outcome == predictedLabel"))
    val countCorrectPredictions = correctPredictions.groupBy("outcome").agg(count("outcome").alias("Correct"))
    countCorrectPredictions.toDF
  }

  def wrongPredictions(predictions: org.apache.spark.sql.Dataset[org.apache.spark.sql.Row]): DataFrame = {
    val wrongPredictions = predictions.where(expr("outcome != predictedLabel"))
    val countErrors = wrongPredictions.groupBy("outcome").agg(count("outcome").alias("Error"))
    countErrors.toDF
  }
  def accuracy(correctPredictions:DataFrame ,wrongPredictions:DataFrame,testData : DataFrame):(Double,String)=
  {
    val totalRecords=testData.count()
    val trueNegative = correctPredictions.where(col("outcome") ===  "live").select("Correct")
    var tn: Long = 0
    if(trueNegative.count() == 0 ){
      tn = 0
    } else
    {
      tn = trueNegative.head().getLong(0)
    }
    var conf = "True Negative : " + tn + "\n"
    val truePositive = correctPredictions.where(col("outcome") ===  "dead").select("Correct")
    var tp: Long = 0
    if(truePositive.count() == 0 ){
      tp = 0
    } else {
      tp = truePositive.head().getLong(0)
    }
    conf += "True Positive : " + tp + "\n"

    val falseNegative = wrongPredictions.where(col("outcome") ===  "dead").select("Error")
    var fn: Long = 0
    if(falseNegative.count() == 0 ){
      fn = 0
    } else
    {
      fn = falseNegative.head().getLong(0)
    }
    conf += "False Negative : " + fn + "\n"
    val falsePositive = wrongPredictions.where(col("outcome") ===  "live").select("Error")
    var fp: Long = 0
    if(falsePositive.count() == 0 ){
      tp = 0
    } else {
      fp = falsePositive.head().getLong(0)
    }
    conf += "False Positive : " + fp+ "\n"

    val accuracy = (tp.toDouble + tn.toDouble)/totalRecords
    (accuracy,conf)
  }
}
