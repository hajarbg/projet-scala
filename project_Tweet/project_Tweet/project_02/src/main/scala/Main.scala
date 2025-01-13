import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{RandomForestClassifier, LogisticRegression}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{HashingTF, NGram, StopWordsRemover, StringIndexer, Tokenizer}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, SparkSession}
import java.util.regex.Pattern
import org.apache.spark.sql.SaveMode

object Main {
  def main(args: Array[String]): Unit = {
    // Initialisation de Spark
    val spark = SparkSession.builder()
      .appName("Advanced Sentiment Analysis")
      .master("local[*]") // Mode local
      .getOrCreate()

    import spark.implicits._

    // UDFs pour prétraiter les données
    val removePunctuationAndSpecialChar = udf { (text: String) =>
      val regex = "[\\.\\,\\:\\-\\!\\?\\n\\t,\\%\\#\\*\\|\\=\\(\\)\\\"\\>\\<\\/\\'\\`\\&\\{\\}\\;\\+\\-\\[\\]\\_\\@]"
      val pattern = Pattern.compile(regex)
      val matcher = pattern.matcher(text)
      matcher.replaceAll(" ").split("[ ]+").mkString(" ")
    }

    val toLowerCase = udf { (text: String) => text.toLowerCase }

    // Charger les données
    def loadData(filename: String): DataFrame = {
      val data = spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv(filename)
        .select("airline_sentiment", "text")
        .toDF("sentiment", "text")
        .na.drop()

      data.withColumn("textClean", removePunctuationAndSpecialChar(toLowerCase($"text")))
    }

    // Charger le fichier CSV
    val filename = "src/main/resources/Tweets.csv"
    val rawData = loadData(filename)

    // Vérification de la distribution des données
    rawData.groupBy("sentiment").count().show()

    // Rééquilibrage des données
    val positive = rawData.filter($"sentiment" === "positive")
    val negative = rawData.filter($"sentiment" === "negative")
    val neutral = rawData.filter($"sentiment" === "neutral")

    val balancedData = positive.union(negative.sample(1.0)).union(neutral.sample(1.0))
    balancedData.groupBy("sentiment").count().show()

    // Prétraitement : Tokenizer et suppression des stopwords
    val tokenizer = new Tokenizer()
      .setInputCol("textClean")
      .setOutputCol("words")

    val remover = new StopWordsRemover()
      .setInputCol("words")
      .setOutputCol("filtered")

    // Ajout des n-grammes
    val ngram = new NGram()
      .setInputCol("filtered")
      .setOutputCol("ngrams")

    // HashingTF
    val hashingTF = new HashingTF()
      .setInputCol("ngrams")
      .setOutputCol("features")

    // StringIndexer
    val indexer = new StringIndexer()
      .setInputCol("sentiment")
      .setOutputCol("label")

    // Modèle avancé : Random Forest
    val rf = new RandomForestClassifier()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setNumTrees(50)

    // Pipeline
    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, remover, ngram, hashingTF, indexer, rf))

    // ParamGridBuilder
    val paramGrid = new ParamGridBuilder()
      .addGrid(hashingTF.numFeatures, Array(1000, 5000, 10000))
      .addGrid(rf.numTrees, Array(10, 50, 100))
      .addGrid(rf.maxDepth, Array(5, 10, 20))
      .build()

    // Évaluateur multi-classes
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")

    // Validation croisée
    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(5)

    // Split des données
    val Array(training, test) = balancedData.randomSplit(Array(0.8, 0.2), seed = 42)

    // Entraînement
    val cvModel = cv.fit(training)

    // Prédictions
    val results = cvModel.transform(test)

    // Évaluation
    val accuracy = evaluator.evaluate(results)
    println(s"Accuracy = $accuracy")

    // Matrice de confusion complète
    results.groupBy("label", "prediction").count().show()

    // Analyse des prédictions incorrectes
    results.filter($"label" =!= $"prediction").select("label", "prediction", "text").show(10, truncate = false)

    // Calcul des métriques
    val resultRDD = results.select("label", "prediction").as[(Double, Double)].rdd
    val metrics = new org.apache.spark.mllib.evaluation.MulticlassMetrics(resultRDD)

    println("Confusion matrix:")
    println(metrics.confusionMatrix)

    metrics.labels.foreach { label =>
      println(s"Precision($label) = ${metrics.precision(label)}")
      println(s"Recall($label) = ${metrics.recall(label)}")
      println(s"F1-Score($label) = ${metrics.fMeasure(label)}")
    }


    //Sentiment Percentage Calculation and CSV saving
    val sentimentCounts = results.groupBy("sentiment").count()
    val totalSentimentCount = sentimentCounts.select(sum("count")).first().getLong(0)

    val sentimentPercentages = sentimentCounts.withColumn("percentage", col("count").cast("double") * 100 / totalSentimentCount)

    sentimentPercentages.select("sentiment", "percentage").write.mode(SaveMode.Overwrite).option("header", "true").csv("sentiment_percentages.csv")


    // Arrêter Spark
    spark.stop()
  }
}