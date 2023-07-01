package com.yasmin.hcp;

import java.io.File;
import java.util.Enumeration;
import java.util.Random;

import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.evaluation.ConfusionMatrix;
import weka.classifiers.evaluation.Evaluation;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Attribute;
import weka.core.SerializationHelper;
import weka.core.converters.CSVLoader;
import weka.core.converters.CSVSaver;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.filters.unsupervised.attribute.StringToNominal;
import weka.filters.unsupervised.attribute.StringToWordVector;

public class WekaTest {

	public static void main(String[] args) throws Exception {

		System.out.println("Weka loaded successfully.");
		
		String fileName = "src/main/resources/datasets/train.csv";
		
		CSVLoader loader = new CSVLoader();
		loader.setFieldSeparator(",");
		loader.setSource(new File(fileName));
		loader.setMissingValue("?");
		Instances instances = loader.getDataSet();

		System.out.println("#1------>"+instances.toSummaryString());
		//instances.deleteAttributeAt(instances.attribute("DEVICETYPE").index());
		instances.deleteAttributeAt(instances.attribute("ID").index());
		//instances.deleteAttributeAt(instances.attribute("PLATFORM_ID").index());
		//instances.deleteAttributeAt(instances.attribute("BIDREQUESTIP").index());
		//instances.deleteAttributeAt(instances.attribute("USERPLATFORMUID").index());
		//instances.deleteAttributeAt(instances.attribute("USERCITY").index());
		//instances.deleteAttributeAt(instances.attribute("USERAGENT").index());
		//instances.deleteAttributeAt(instances.attribute("CHANNELTYPE").index());
		
		NumericToNominal numericToNominal = new NumericToNominal();
		

		numericToNominal.setInputFormat(instances);
		instances = Filter.useFilter(instances,numericToNominal);
		System.out.println("#2------>"+instances.toSummaryString());
		instances.setClassIndex(instances.attribute("IS_HCP").index());
		instances.randomize(new Random(23));
		
		int trainSize = (int) Math.round(instances.numInstances()*0.80);
		int testSize = instances.numInstances() - trainSize;
		
		Instances trainInstances = new Instances(instances,0,trainSize);
		Instances testInstances = new Instances(instances,trainSize,testSize);
		
		int lastIndex = trainInstances.numAttributes() - 1;
		
		trainInstances.setClass(instances.attribute("IS_HCP"));
		testInstances.setClass(instances.attribute("IS_HCP"));
		
		StringToWordVector filter = new StringToWordVector();
		filter.setInputFormat(trainInstances);
		
		trainInstances = Filter.useFilter(trainInstances, filter);
		testInstances = Filter.useFilter(testInstances, filter);
		
		NaiveBayes naiveBayes = new NaiveBayes();
		naiveBayes.buildClassifier(trainInstances);
		
		Evaluation evaluation = new Evaluation(testInstances);
		evaluation.evaluateModel(naiveBayes,testInstances);
		//CSVSaver predictedtCsvSaver = new CSVSaver();
		//predictedtCsvSaver.setFile(new File("src/main/resources/datasets/testingpredict.csv"));
		//predictedtCsvSaver.setInstances(testInstances);
		//predictedtCsvSaver.writeBatch();
		
		
		
		System.out.println(evaluation.toSummaryString());
		
		ConfusionMatrix confusionMatrix = new ConfusionMatrix(new String[] {"hcp","not-hcp"});
		confusionMatrix.addPredictions(evaluation.predictions());
		
		System.out.println(confusionMatrix.toString());
		
		System.out.println("Accuracy: "+evaluation.pctCorrect());
		System.out.println("Precision: "+evaluation.precision(1));
		System.out.println("Recall: "+evaluation.recall(1));
		
		SerializationHelper.write("hcp.model", naiveBayes);
		System.out.println("Saved trained model to hcp.model ----> ");
		NaiveBayes nb =  (NaiveBayes) SerializationHelper
				.read("hcp.model");
		
		String fileNameTest = "src/main/resources/datasets/test.csv";
		loader = new CSVLoader();
		loader.setFieldSeparator(",");
		loader.setSource(new File(fileNameTest));
		loader.setMissingValue("?");
		Instances predictInstances = loader.getDataSet();
		System.out.println("#test1------>"+predictInstances.toSummaryString());

		//predictInstances.deleteAttributeAt(predictInstances.attribute("DEVICETYPE").index());
		predictInstances.deleteAttributeAt(predictInstances.attribute("ID").index());
		//predictInstances.deleteAttributeAt(predictInstances.attribute("PLATFORM_ID").index());
		//predictInstances.deleteAttributeAt(predictInstances.attribute("BIDREQUESTIP").index());
		//predictInstances.deleteAttributeAt(predictInstances.attribute("USERPLATFORMUID").index());
		//predictInstances.deleteAttributeAt(predictInstances.attribute("USERCITY").index());
		//predictInstances.deleteAttributeAt(predictInstances.attribute("USERAGENT").index());
		//predictInstances.deleteAttributeAt(predictInstances.attribute("CHANNELTYPE").index());
		numericToNominal = new NumericToNominal();
		numericToNominal.setInputFormat(predictInstances);
		predictInstances = Filter.useFilter(predictInstances,numericToNominal);
		
		System.out.println("#test2------>"+predictInstances.toSummaryString());
		predictInstances.setClass(predictInstances.attribute("IS_HCP"));

		CSVLoader p1Loader = new CSVLoader();
		p1Loader.setSource(new File("src/main/resources/datasets/test.csv"));
		Instances predict1Instances = p1Loader.getDataSet();
		predict1Instances.deleteAttributeAt(predict1Instances.attribute("DEVICETYPE").index());
		predict1Instances.deleteAttributeAt(predict1Instances.attribute("USERZIPCODE").index());
		predict1Instances.deleteAttributeAt(predict1Instances.attribute("PLATFORM_ID").index());
		predict1Instances.deleteAttributeAt(predict1Instances.attribute("BIDREQUESTIP").index());
		predict1Instances.deleteAttributeAt(predict1Instances.attribute("USERPLATFORMUID").index());
		predict1Instances.deleteAttributeAt(predict1Instances.attribute("USERCITY").index());
		predict1Instances.deleteAttributeAt(predict1Instances.attribute("USERAGENT").index());
		predict1Instances.deleteAttributeAt(predict1Instances.attribute("CHANNELTYPE").index());
		predict1Instances.deleteAttributeAt(predict1Instances.attribute("PLATFORMTYPE").index());
		predict1Instances.deleteAttributeAt(predict1Instances.attribute("URL").index());
		predict1Instances.deleteAttributeAt(predict1Instances.attribute("KEYWORDS").index());
		predict1Instances.deleteAttributeAt(predict1Instances.attribute("TAXONOMY").index());
		predict1Instances.setClass(predict1Instances.attribute("IS_HCP"));
		
		Enumeration pInstances = predictInstances.enumerateInstances();
		Enumeration p1Instances = predict1Instances.enumerateInstances();
		while (pInstances.hasMoreElements()) {
			Instance instance = (Instance) pInstances.nextElement();
			Instance instance1 = (Instance) p1Instances.nextElement();
			double classification = nb.classifyInstance(instance);
			instance1.setClassValue(classification);
		}

		
		CSVSaver predictedCsvSaver = new CSVSaver();
		predictedCsvSaver.setFile(new File("src/main/resources/datasets/predict.csv"));
		predictedCsvSaver.setInstances(predict1Instances);
		predictedCsvSaver.writeBatch();

		System.out.println("Prediciton saved to predict.csv");
		

	}

}
