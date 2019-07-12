package com.amicabile.spamclassifier;


import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.feature.Tokenizer;
import org.apache.spark.ml.feature.HashingTF;

import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructType;

import java.io.File;

import static org.apache.spark.sql.functions.col;


public class CreateModel {

    public static void main(String[] args) {

        File directory = new File("data");
        if (!directory.exists()) {
            directory.mkdir();
        }

        SparkSession spark = SparkSession
                .builder()
                .appName("Java Spark SQL Example")
                .config("spark.master", "local")
                .config("spark.testing.memory", "2147480000")
                .getOrCreate();

        JavaSparkContext jsc = new JavaSparkContext(spark.sparkContext());
        jsc.setLogLevel("WARN");

        StructType schema = new StructType()
                .add("message", "string")
                .add("label", "int");


        Dataset<Row> df = spark.read()
                .option("mode", "DROPMALFORMED")
                .schema(schema)
                .csv("spam_out.csv");

        Dataset<Row>[] splits = df.randomSplit(new double[] {0.8, 0.2}, 42);
        Dataset<Row> trainingDf = splits[0];
        Dataset<Row> testDf = splits[1];


        Tokenizer tokenizer = new Tokenizer()
                .setInputCol("message")
                .setOutputCol("words");
        //CountVectorizer countVectorizer = new CountVectorizer().setInputCol(tokenizer.getOutputCol()).setOutputCol("features");
        HashingTF hashingTF = new HashingTF()
                .setNumFeatures(3000)
                .setInputCol(tokenizer.getOutputCol())
                .setOutputCol("features");
        LogisticRegression lr = new LogisticRegression()
                .setMaxIter(100).setRegParam(0.001);
        Pipeline pipeline = new Pipeline()
                .setStages(new PipelineStage[] {tokenizer, hashingTF, lr});
        PipelineModel model = pipeline.fit(trainingDf);


        try {
            model.write().overwrite().save("data/sparkmodel");
        } catch (java.io.IOException ioe) {
            System.out.format("Cannot save model to pipeline.model: %s\n", ioe.getMessage());
        }

        Dataset<Row> testPredictions = model.transform(testDf).withColumn("label", testDf.col("label").cast(DataTypes.DoubleType));


        Dataset<Row> testPredictionsRDD = testPredictions.select(col("prediction"), col("label"));
        BinaryClassificationMetrics testMetrics = new BinaryClassificationMetrics(testPredictionsRDD);

        System.out.format("Test PR and ROC : %f, %f\n", testMetrics.areaUnderPR(), testMetrics.areaUnderROC()) ;


        MulticlassMetrics testMulticlassMetrics
                = new MulticlassMetrics(testPredictionsRDD);

// Confusion matrix

        System.out.println("Confusion Matrix");
        System.out.println(testMulticlassMetrics.confusionMatrix());
        System.out.format("Precision: %f, %f\n",testMulticlassMetrics.precision(0), testMulticlassMetrics.precision(1));
        System.out.format("Recall: %f, %f\n",testMulticlassMetrics.recall(0), testMulticlassMetrics.recall(1));






    }


}
