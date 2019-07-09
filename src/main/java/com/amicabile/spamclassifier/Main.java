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
import org.apache.spark.ml.feature.Tokenizer;
import org.apache.spark.ml.feature.HashingTF;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SQLContext;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.StructType;


import org.apache.spark.ml.feature.RegexTokenizer;
import org.apache.spark.ml.feature.Tokenizer;
import sun.misc.JavaLangAccess;


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

        df.show();

        Tokenizer tokenizer = new Tokenizer()
                .setInputCol("message")
                .setOutputCol("words");
        HashingTF hashingTF = new HashingTF()
                .setNumFeatures(1000)
                .setInputCol(tokenizer.getOutputCol())
                .setOutputCol("features");
        LogisticRegression lr = new LogisticRegression()
                .setMaxIter(10)
                .setRegParam(0.001);
        Pipeline pipeline = new Pipeline()
                .setStages(new PipelineStage[] {tokenizer, hashingTF, lr});
        PipelineModel model = pipeline.fit(df);

        Dataset<Row> predictions = model.transform(df);
        for (Row r : predictions.select( "message", "probability", "prediction").collectAsList()) {
            System.out.println("(" + r.get(0)  + ") --> prob=" + r.get(1)
                    + ", prediction=" + r.get(2));
        }


    }
}
