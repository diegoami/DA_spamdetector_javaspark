package com.amicabile.spamclassifier;

import org.apache.commons.io.LineIterator;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.SQLContext;

import java.util.function.Function;

public class Main {


    static private String stripQuotes(String string, char quoteChar){

        if(string.length() > 1 && ((string.charAt(0) == quoteChar) && (string.charAt(string.length() - 1) == quoteChar))){
            return string.substring(1, string.length() - 1);
        }

        return string;
    }


    public static void main(String[] args) {

        SparkConf conf = new SparkConf()
                .setAppName("Word Count").setMaster("local");
        JavaSparkContext sc = new JavaSparkContext(conf);
        JavaRDD<String> data = sc.textFile("spam_out.csv");
        //JavaSQLContext sqlContext = new JavaSQLContext(sc); // For previous versions
        SQLContext sqlContext = new SQLContext(sc); // In Spark 1.3 the Java API and Scala API have been unified


        JavaRDD<Record> rdd_records = data.map(
                line -> {
                    try {
                        String message = stripQuotes(line.substring(0, line.length() - 2), '"');

                        String labelString = line.substring(line.length() - 1, line.length());
                        int label = Integer.parseInt(labelString );
                        System.out.format("Processing row %s\n", message);
                        return new Record(message, label);
                    } catch (NumberFormatException nfe) {
                        System.out.format("Skipping row  %s\n", nfe.getMessage());
                        return new Record("", 0);
                    }
                }
        );


    }
}
