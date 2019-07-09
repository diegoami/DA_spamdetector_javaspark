package com.amicabile.spamclassifier;

import org.apache.commons.io.LineIterator;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.Tokenizer;
import org.apache.spark.ml.feature.HashingTF;
import org.apache.spark.ml.feature.CountVectorizer;

import org.apache.spark.ml.tuning.TrainValidationSplit;
import org.apache.spark.ml.tuning.TrainValidationSplitModel;
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SQLContext;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructType;

import static org.apache.spark.sql.functions.col;



public class Main {

    public static void main(String[] args) {

        SparkSession spark = SparkSession
                .builder()
                .appName("Java Spark SQL Example")
                .config("spark.master", "local")
                .getOrCreate();

        StructType schema = new StructType()
                .add("message", "string")
                .add("label", "int");


        Dataset<Row> df = spark.read()
                .option("mode", "DROPMALFORMED")
                .schema(schema)
                .csv("spam_out.csv");
                //.map(line -> new JavaLabeledDocument(0L, line[0], Double.parseDouble(line[1])));

        Dataset<Row>[] splits = df.randomSplit(new double[] {0.8, 0.2}, 12345);
        Dataset<Row> trainingDf = splits[0];
        Dataset<Row> testDf = splits[1];


        Tokenizer tokenizer = new Tokenizer()
                .setInputCol("message")
                .setOutputCol("words");
        CountVectorizer countVectorizer = new CountVectorizer().setInputCol(tokenizer.getOutputCol())
                .setOutputCol("features");
        HashingTF hashingTF = new HashingTF()
                .setNumFeatures(3000)
               .setInputCol(tokenizer.getOutputCol())
                .setOutputCol("features");
        LogisticRegression lr = new LogisticRegression()
                .setMaxIter(100).setRegParam(0.001);
        Pipeline pipeline = new Pipeline()
                .setStages(new PipelineStage[] {tokenizer, hashingTF, lr});
        PipelineModel model = pipeline.fit(trainingDf);

        Dataset<Row> testPredictions = model.transform(testDf).withColumn("label", testDf.col("label").cast(DataTypes.DoubleType));

        Dataset<Row> predictions = model.transform(df).withColumn("label", df.col("label").cast(DataTypes.DoubleType));
        //for (Row r : predictions.select( "message", "probability", "prediction").collectAsList()) {
        //    System.out.println("(" + r.get(0)  + ") --> prob=" + r.get(1)
        //            + ", prediction=" + r.get(2));
        //}


        BinaryClassificationMetrics testMetrics =
                new BinaryClassificationMetrics(testPredictions.select(col("prediction"), col("label")));

        System.out.format("Test PR and ROC : %f, %f\n", testMetrics.areaUnderPR(), testMetrics.areaUnderROC()) ;
        BinaryClassificationMetrics overallMetrics =
                new BinaryClassificationMetrics(predictions.select(col("prediction"), col("label")));

        System.out.format("Overall PR and ROC : %f, %f\n", overallMetrics.areaUnderPR(), overallMetrics .areaUnderROC());

        MulticlassMetrics metrics = new MulticlassMetrics(predictions.select("prediction", "label"));

// Confusion matrix
        Matrix confusion = metrics.confusionMatrix();
        System.out.println("Confusion matrix: \n" + confusion);


    }
}
