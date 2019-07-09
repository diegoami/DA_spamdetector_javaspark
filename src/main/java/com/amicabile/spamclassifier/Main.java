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
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import java.util.Arrays;
import java.util.List;

import static org.apache.spark.sql.functions.col;


public class Main {

    public static void main(String[] args) {

        SparkSession spark = SparkSession
                .builder()
                .appName("Java Spark SQL Example")
                .config("spark.master", "local")
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
        try {
            pipeline.write().overwrite().save("pipeline");
            model.write().overwrite().save("model");
        } catch (java.io.IOException ioe) {
            System.out.format("Cannot save model to pipeline.model: %s\n", ioe.getMessage());
        }

        PipelineModel loadedModel = PipelineModel.load("model");
        List<Row> rowList = Arrays.asList(
                RowFactory.create("Winner! You have won a car"),
                RowFactory.create("I feel bad today"),
                RowFactory.create("Please call our customer service representative"),
                RowFactory.create("Your free ringtone is waiting to be collected. Simply text the password")
        );

        StructType schemaPred = new StructType()
                .add("message", "string");

        Dataset<Row> rowDf = spark.createDataFrame(rowList, schemaPred);
        Dataset<Row> predictionsLoaded = loadedModel.transform(rowDf);
        System.out.println(predictionsLoaded);
        for (Row r : predictionsLoaded.select( "message", "probability", "prediction").collectAsList()) {
            System.out.println("(" + r.get(0)  + ") --> prob=" + r.get(1)
                    + ", prediction=" + r.get(2));
        }
    }
}
