package com.amicabile.spamclassifier;


import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.feature.HashingTF;
import org.apache.spark.ml.feature.Tokenizer;
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructType;

import java.util.Arrays;
import java.util.List;

import static org.apache.spark.sql.functions.col;


public class TestModel {

    public static void main(String[] args) {

        SparkSession spark = SparkSession
                .builder()
                .appName("Java Spark SQL Example")
                .config("spark.master", "local")
                .getOrCreate();

        JavaSparkContext jsc = new JavaSparkContext(spark.sparkContext());
        jsc.setLogLevel("WARN");


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
