package com.yasmin.hcp;

import java.io.File;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.evaluation.Evaluation;
import weka.clusterers.ClusterEvaluation;
import weka.clusterers.SimpleKMeans;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.core.converters.CSVSaver;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.filters.unsupervised.attribute.StringToWordVector;


public class WekaTaxonomy {

	public static void main(String[] args) throws Exception {
		
		String fileName = "src/main/resources/datasets/train.csv";
		
		CSVLoader loader = new CSVLoader();
		loader.setFieldSeparator(",");
		loader.setSource(new File(fileName));
		//loader.setMissingValue("?");
		Instances instances = loader.getDataSet();
		instances.deleteWithMissing(instances.attribute("TAXONOMY"));
		System.out.println("#1------>"+instances.toSummaryString());
		instances.deleteAttributeAt(instances.attribute("DEVICETYPE").index());
		instances.deleteAttributeAt(instances.attribute("ID").index());
		instances.deleteAttributeAt(instances.attribute("PLATFORM_ID").index());
		instances.deleteAttributeAt(instances.attribute("BIDREQUESTIP").index());
		instances.deleteAttributeAt(instances.attribute("USERPLATFORMUID").index());
		instances.deleteAttributeAt(instances.attribute("USERCITY").index());
		instances.deleteAttributeAt(instances.attribute("USERAGENT").index());
		instances.deleteAttributeAt(instances.attribute("CHANNELTYPE").index());
		
		NumericToNominal numericToNominal = new NumericToNominal();
		

		numericToNominal.setInputFormat(instances);
		instances = Filter.useFilter(instances,numericToNominal);
		System.out.println("#2------>"+instances.toSummaryString());

		
		SimpleKMeans simpleKMeans = new SimpleKMeans();

		// build clusterer
		simpleKMeans.setPreserveInstancesOrder(true);
		simpleKMeans.setNumClusters(207);
		simpleKMeans.buildClusterer(instances);

		ClusterEvaluation eval = new ClusterEvaluation();
		eval.setClusterer(simpleKMeans);
		eval.evaluateClusterer(instances);

		System.out.println("Cluster Evaluation: "+eval.clusterResultsToString());
		int index =0;
		int[] assignedClusters = new int[instances.numInstances()];
		
		for (Instance instance : instances) {
			assignedClusters[index++] = simpleKMeans.clusterInstance(instance);
		}
		
		Attribute classAttribute = new Attribute ("taxClass",false);
		
		instances.insertAttributeAt(classAttribute,instances.numAttributes());
		
		index =0;
		int attributeIndex = instances.numAttributes()-1;
		
		for (Instance instance : instances) {
			instance.setValue(attributeIndex,assignedClusters[index++]);
		}
		
		NumericToNominal numericToNom = new NumericToNominal();
		numericToNom.setInputFormat(instances);
		
		Instances processedInstances = Filter.useFilter(instances,numericToNom);
		
		System.out.println(processedInstances.toSummaryString());

		Map<Double,String> taxClassNomy = new HashMap<Double,String>();
		Enumeration mapInstances = processedInstances.enumerateInstances();
		while (mapInstances.hasMoreElements()) {
			Instance instance = (Instance) mapInstances.nextElement();
			taxClassNomy.put(instance.value(instances.attribute("taxClass")),instance.stringValue(instances.attribute("TAXONOMY")));
		}

		
		processedInstances.setClassIndex(instances.attribute("taxClass").index());
		
		processedInstances.randomize(new Random(23));
		
		int trainSize = (int) Math.round(processedInstances.numInstances()*0.80);
		int testSize = processedInstances.numInstances() - trainSize;
		
		Instances trainInstances = new Instances(processedInstances,0,trainSize);
		Instances testInstances = new Instances(processedInstances,trainSize,testSize);
		
		
		trainInstances.setClass(instances.attribute("taxClass"));
		testInstances.setClass(instances.attribute("taxClass"));
		
		StringToWordVector filter = new StringToWordVector();
		filter.setInputFormat(trainInstances);
		
		trainInstances = Filter.useFilter(trainInstances, filter);
		testInstances = Filter.useFilter(testInstances, filter);
		
		NaiveBayes naiveBayes = new NaiveBayes();
		naiveBayes.buildClassifier(trainInstances);
		
		Evaluation evaluation = new Evaluation(testInstances);
		evaluation.evaluateModel(naiveBayes,testInstances);
	
		System.out.println(evaluation.toSummaryString());


		
		System.out.println("Accuracy: "+evaluation.pctCorrect());
		System.out.println("Precision: "+evaluation.precision(1));
		System.out.println("Recall: "+evaluation.recall(1));

		
		String fileNameTest = "src/main/resources/datasets/predictForTax.csv";
		loader = new CSVLoader();
		loader.setFieldSeparator(",");
		loader.setSource(new File(fileNameTest));
		loader.setMissingValue("?");
		Instances predictInstances = loader.getDataSet();
		
//		System.out.println("#test1------>"+predictInstances.toSummaryString());

		predictInstances.deleteAttributeAt(predictInstances.attribute("ID").index());
		predictInstances.deleteAttributeAt(predictInstances.attribute("TAXONOMY").index());
		Attribute taxClass = new Attribute("taxClass");
		predictInstances.insertAttributeAt(taxClass,predictInstances.numAttributes());
		numericToNominal = new NumericToNominal();
		numericToNominal.setInputFormat(predictInstances);
		predictInstances = Filter.useFilter(predictInstances,numericToNominal);
		
//		System.out.println("#test2------>"+predictInstances.toSummaryString());
		predictInstances.setClass(predictInstances.attribute("taxClass"));

		CSVLoader p1Loader = new CSVLoader();
		p1Loader.setSource(new File("src/main/resources/datasets/predictForTax.csv"));
		Instances predict1Instances = p1Loader.getDataSet();
		predict1Instances.deleteAttributeAt(predict1Instances.attribute("USERZIPCODE").index());
		predict1Instances.deleteAttributeAt(predict1Instances.attribute("PLATFORMTYPE").index());
		predict1Instances.deleteAttributeAt(predict1Instances.attribute("URL").index());
		predict1Instances.deleteAttributeAt(predict1Instances.attribute("KEYWORDS").index());
		predict1Instances.deleteAttributeAt(predict1Instances.attribute("TAXONOMY").index());
		taxClass = new Attribute("taxClass");
		predict1Instances.insertAttributeAt(taxClass,predict1Instances.numAttributes());
		predict1Instances.setClass(predict1Instances.attribute("taxClass"));
		
		Evaluation evaluationT = new Evaluation(predictInstances);
		evaluationT.evaluateModel(naiveBayes,predictInstances);
		Attribute taxonomy = new Attribute("TAXONOMY",true);
		predict1Instances.insertAttributeAt(taxonomy,predict1Instances.numAttributes());

		int taxonomyP = predict1Instances.attribute("TAXONOMY").index(); 
		
		Enumeration pInstances = predictInstances.enumerateInstances();
		Enumeration p1Instances = predict1Instances.enumerateInstances();
		while (pInstances.hasMoreElements()) {
			Instance instance = (Instance) pInstances.nextElement();
			Instance instance1 = (Instance) p1Instances.nextElement();
		
			Double classification = instance.value(predict1Instances.attribute("taxClass"));
			instance1.setClassValue(classification);
			String predictedTaxonomy = taxClassNomy.get(classification);
			if(predictedTaxonomy!=null)
			instance1.setValue(taxonomyP,predictedTaxonomy);
		}
		CSVSaver predictedCsvSaver = new CSVSaver();
		predictedCsvSaver.setFile(new File("src/main/resources/datasets/predictTaxonomy.csv"));
		predictedCsvSaver.setInstances(predict1Instances);
		predictedCsvSaver.writeBatch();

		System.out.println("Prediciton saved to predictTaxonomy.csv");

	}

}