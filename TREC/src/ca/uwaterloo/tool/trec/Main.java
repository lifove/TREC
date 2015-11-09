package ca.uwaterloo.tool.trec;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;

import hk.ust.cse.ipam.utils.DecimalUtil;
import hk.ust.cse.ipam.utils.HashMapUtil;
import hk.ust.cse.ipam.utils.WekaUtils;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.Utils;

public class Main {

	/**
	 * @param args
	 * 
	 * TREC mode: Tool / Experimental Model
	 * Options for Experimental Mode: n-fold cross validation
	 * 
	 */
	public static void main(String[] args) {
		
		new Main().run(args);
		
	}
	
	
	String file;
	String lableInfo;
	String source;
	Boolean experimental;
	int repeat =1;
	int folds = 2;
	String mlAlgorithm = "weka.classifiers.functions.Logistic";
	int currentFold = 0;

	private void run(String[] args) {
		
		processOptions(args);
		
		// load a test set
		Instances instances = WekaUtils.loadArff(file);
		
		// load a pool of sources, exclude the test set
		HashMap<String,Instances> sources = WekaUtils.loadArffs(source,file);
		
		// if experimental, repeat n-fold cross validation for 
		if(experimental){
			for(int i=0; i<repeat; i++){
				
				// randomize with different seed for each iteration
				instances.randomize(new Random(i)); 
				instances.stratify(folds);
				
				for(int n=0;n < folds;n++){
					
					Instances tarTrain = instances.trainCV(folds, n);
					Instances tarTest = instances.testCV(folds, n);
					
					sources.put(new File(file).getName(), tarTrain);
					
					currentFold = n;
					
					generateTRECTable(sources,tarTest);
					
				}
			}
		}else{
			generateTRECTable(sources,instances);
		}
	}
	
	private void generateTRECTable(HashMap<String,Instances> sources,Instances instances){
		
		// preprocessing
		instances = preprocessing(instances);
		
		// identify the best sources
		DatasetAnalyzer da = new DatasetAnalyzer(instances,sources);
		da.analyze();
		HashMap<String,Double> scores = da.similarityScores;
		
		// sort
		scores = (HashMap<String, Double>) HashMapUtil.sortByValue(scores);
		
		// build a model using each similar src dataset and test the model on the target dataset
		// compute precision-recall curve, AUCEC
		// generate TREC table
		for(String srcFile:scores.keySet()){
			
			// consider only average score is > 0. Others will have > 0.05 since KS cutoff is 0.05
			// TODO just show all results for now
			//if(scores.get(srcFile)<0)
			//	continue;
			
			// (0) target file (1) fold index
			System.out.print(file + "," + (currentFold+1) + ",");
			
			// (2)src file and (3) similarity score betwen src and tar.
			System.out.print(srcFile + "," + DecimalUtil.threeDecimal(scores.get(srcFile)) +",");
			
			// generate new src and tar datasets using matched attributes
			
			// (1) get a list of src and tar attribute indice matched
			
			ArrayList<Integer> srcSelectedIndice = new ArrayList<Integer>();
			ArrayList<Integer> tarSelectedIndice = new ArrayList<Integer>();
			for(String matchedAttribute:da.allFinallyMatchedAttributes.get(srcFile).keySet()){
				
				String[] indices = matchedAttribute.split("-");
				
				srcSelectedIndice.add(Integer.parseInt(indices[0]));
				tarSelectedIndice.add(Integer.parseInt(indices[1]));
				
			}
			
			Instances src = sources.get(srcFile.replace("(H)","")); // to get the original data
			src = WekaUtils.getInstancesWithSelectedAttributes(src, srcSelectedIndice, WekaUtils.getPosLabel(src));
			Instances tar = WekaUtils.getInstancesWithSelectedAttributes(instances, tarSelectedIndice, WekaUtils.getPosLabel(instances));
			
			// build model
			int posClassValueIndex = WekaUtils.getClassValueIndex(tar, WekaUtils.strPos);
			Classifier classifier;
			try {
				classifier = (Classifier) Utils.forName(Classifier.class, mlAlgorithm, null);
				Classifier clsCopy = AbstractClassifier.makeCopy(classifier);
				clsCopy.buildClassifier(src);
				
				// evaluate the model on itself to get precision-recall curve
				Evaluation evalForSrc = new Evaluation(src);
				evalForSrc.evaluateModel(clsCopy, src);
				Instances srcCurve = WekaUtils.getCurve(evalForSrc.predictions(), posClassValueIndex);
				
				// get all thresholds from src prediction results
				double[] srcThresholds = srcCurve.attributeToDoubleArray(srcCurve.attribute("Threshold").index());
				
				ArrayList<String> predictionResults = WekaUtils.getPrecisionRecallFmeasureFromCurve(srcCurve);
				
				// evaluate the model on the target set
				Evaluation evalForTar = new Evaluation(src);
				evalForTar.evaluateModel(clsCopy, tar);
				Instances tarCurve = WekaUtils.getCurve(evalForTar.predictions(), posClassValueIndex);
				ArrayList<String> tarPredictionResults = WekaUtils.getPrecisionRecallFmeasureFromCurve(tarCurve);
				
				double[] tarThreholds = tarCurve.attributeToDoubleArray(tarCurve.attribute("Threshold").index());
				
				// compare srcThresholds and tarThresholds using KS-test
				// if their threshold distributions are not similar, skip this prediction
				double thresholdSimilarity = da.getKSPvalueFromR(srcThresholds, tarThreholds);
				/*if(thresholdSimilarity<=0.05){
					continue;
				}*/
				
				// (3.5) threshold similarity
				System.out.print(DecimalUtil.threeDecimal(thresholdSimilarity) + ",");
				
				// compute src's precision recall curve and AUCEC
				//System.out.println("Source best results by\n P, R, F1, Thresholds");
				
				ThresholdManager tmForSrc = new ThresholdManager(predictionResults);
				
				Measure bestPrecisionResult = tmForSrc.processResultsToGetBestInfo(tmForSrc.IDX_PRECISON);
				Measure bestRecallResult = tmForSrc.processResultsToGetBestInfo(tmForSrc.IDX_RECALL);
				Measure bestF1Result = tmForSrc.processResultsToGetBestInfo(tmForSrc.IDX_F1);
				
				// source best results by the best src Precision
				// (4) Precision(5) Recall (6) F1 (7) Threshold
				System.out.print(DecimalUtil.threeDecimal(bestPrecisionResult.getPrecision()) + "," + 
									DecimalUtil.threeDecimal(bestPrecisionResult.getRecall()) + "," +
									DecimalUtil.threeDecimal(bestPrecisionResult.getF1()) + "," +
									DecimalUtil.threeDecimal(bestPrecisionResult.getThreshold())+",");
				// source best results by the best src Recall
				// (8) Precision (9) Recall (10) F1 (11) Threshold
				System.out.print(DecimalUtil.threeDecimal(bestRecallResult.getPrecision()) + "," + 
									DecimalUtil.threeDecimal(bestRecallResult.getRecall()) + "," +
									DecimalUtil.threeDecimal(bestRecallResult.getF1()) + "," +
									DecimalUtil.threeDecimal(bestRecallResult.getThreshold())+",");
				// source best results by the best src F1
				// (12) Precision (13) Recall (14) F1 (15) Threshold
				System.out.print(DecimalUtil.threeDecimal(bestF1Result.getPrecision()) + "," + 
									DecimalUtil.threeDecimal(bestF1Result.getRecall()) + "," +
									DecimalUtil.threeDecimal(bestF1Result.getF1()) + "," +
									DecimalUtil.threeDecimal(bestF1Result.getThreshold()) + ",");
				
				// report results using >= 0.5 threshold
				Measure resultBy50ThdValueforSrc = tmForSrc.getResultByThreshold(0.5);
				
				//System.out.println("Source results when using 0.5 threshold\n P, R, F1, Effective Threshold");
				
				
				//Source results when using 0.5 threshold
				// (16) Precision by the best src precision (17) Recall (18) F1 (19) Threshold
				System.out.print(DecimalUtil.threeDecimal(resultBy50ThdValueforSrc.getPrecision()) + "," + 
						DecimalUtil.threeDecimal(resultBy50ThdValueforSrc.getRecall()) + "," +
						DecimalUtil.threeDecimal(resultBy50ThdValueforSrc.getF1()) + "," +
						DecimalUtil.threeDecimal(resultBy50ThdValueforSrc.getThreshold()) + ",");
				
				// compute tar's precision recall curve and AUCEC
				//System.out.println("***Target results");
				
				ThresholdManager tmforTar = new ThresholdManager(tarPredictionResults);
				
				Measure resultBySrcBestPrecisionUsingThdValue = tmforTar.getResultByThreshold(bestPrecisionResult.getThreshold());
				Measure resultBySrcBestRecallUsingThdValue = tmforTar.getResultByThreshold(bestRecallResult.getThreshold());
				Measure resultBySrcBestF1UsingThdValue = tmforTar.getResultByThreshold(bestF1Result.getThreshold());
				
				//System.out.println("Target results when using the best\n P, R, F1, Thresholds");
				
				// Target results when using the best src Precision
				// (20) Precision by the best src precision (21) Recall (22) F1 (23) Threshold
				System.out.print(DecimalUtil.threeDecimal(resultBySrcBestPrecisionUsingThdValue.getPrecision()) + "," + 
						DecimalUtil.threeDecimal(resultBySrcBestPrecisionUsingThdValue.getRecall()) + "," +
						DecimalUtil.threeDecimal(resultBySrcBestPrecisionUsingThdValue.getF1()) + "," +
						DecimalUtil.threeDecimal(resultBySrcBestPrecisionUsingThdValue.getThreshold()) + ",");
				// Target best results by the best src Recall
				// (24) Precision (25) Recall (26) F1 (27) Threshold
				System.out.print(DecimalUtil.threeDecimal(resultBySrcBestRecallUsingThdValue.getPrecision()) + "," + 
						DecimalUtil.threeDecimal(resultBySrcBestRecallUsingThdValue.getRecall()) + "," +
						DecimalUtil.threeDecimal(resultBySrcBestRecallUsingThdValue.getF1()) + "," +
						DecimalUtil.threeDecimal(resultBySrcBestRecallUsingThdValue.getThreshold()) + ",");
				// Target best results by the best src F1
				// (28) Precision (29) Recall (30) F1 (31) Threshold
				System.out.print(DecimalUtil.threeDecimal(resultBySrcBestF1UsingThdValue.getPrecision()) + "," + 
						DecimalUtil.threeDecimal(resultBySrcBestF1UsingThdValue.getRecall()) + "," +
						DecimalUtil.threeDecimal(resultBySrcBestF1UsingThdValue.getF1()) + "," +
						DecimalUtil.threeDecimal(resultBySrcBestF1UsingThdValue.getThreshold()) + ",");
				
				Measure resultBySrcBestPrecision = tmforTar.getResultByThresholdPercentileRank(bestPrecisionResult.getPercentileRankOfThreshold());
				Measure resultBySrcBestRecall = tmforTar.getResultByThresholdPercentileRank(bestRecallResult.getPercentileRankOfThreshold());
				Measure resultBySrcBestF1 = tmforTar.getResultByThresholdPercentileRank(bestF1Result.getPercentileRankOfThreshold());
				
				//System.out.println("Target results when using percentile ranks of the best\n P, R, F1, Thresholds");
				
				// Target results when using percentile ranks of the best src precision
				// (32) Precision (33) Recall (34) F1 (35) Threshold
				System.out.print(DecimalUtil.threeDecimal(resultBySrcBestPrecision.getPrecision()) + "," + 
						DecimalUtil.threeDecimal(resultBySrcBestPrecision.getRecall()) + "," +
						DecimalUtil.threeDecimal(resultBySrcBestPrecision.getF1()) + "," +
						DecimalUtil.threeDecimal(resultBySrcBestPrecision.getThreshold()) + ",");
				// Target results when using percentile ranks of the best src Recall
				// (36) Precision (37) Recall (38) F1 (39) Threshold
				System.out.print(DecimalUtil.threeDecimal(resultBySrcBestRecall.getPrecision()) + "," + 
						DecimalUtil.threeDecimal(resultBySrcBestRecall.getRecall()) + "," +
						DecimalUtil.threeDecimal(resultBySrcBestRecall.getF1()) + "," +
						DecimalUtil.threeDecimal(resultBySrcBestRecall.getThreshold()) + ",");
				// (40) Precision (41) Recall (42) F1 (43) Threshold
				// Target results when using percentile ranks of the best src F1
				System.out.print(DecimalUtil.threeDecimal(resultBySrcBestF1.getPrecision()) + "," + 
						DecimalUtil.threeDecimal(resultBySrcBestF1.getRecall()) + "," +
						DecimalUtil.threeDecimal(resultBySrcBestF1.getF1()) + "," +
						DecimalUtil.threeDecimal(resultBySrcBestF1.getThreshold()) + ",");
				
				// report results using >= 0.5 threshold
				Measure resultBy50ThdValue = tmforTar.getResultByThreshold(0.5);
				
				//System.out.println("Target results when using 0.5 threshold\n P, R, F1, Effective Threshold");
				
				// Target results when using 0.5 threshold
				// (44) Precision (45) Recall (46) F1 (47) Threshold
				System.out.print(DecimalUtil.threeDecimal(resultBy50ThdValue.getPrecision()) + "," + 
						DecimalUtil.threeDecimal(resultBy50ThdValue.getRecall()) + "," +
						DecimalUtil.threeDecimal(resultBy50ThdValue.getF1()) + "," +
						DecimalUtil.threeDecimal(resultBy50ThdValue.getThreshold()));
				
				System.out.println();
				
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
		
		// generate TREC table
	}

	private Instances preprocessing(Instances instances) {
		
		// feature selection
		
		return instances;
	}

	void processOptions(String[] args){

		// create Options object

		Options options = new Options();

		// add options
		options.addOption("file", true, "Arff file where to predict defects.");
		options.addOption("labelinfo", true, "a file path for label information");
		options.addOption("source", true, "a directory path for all source datasets");
		options.addOption("experimental", false, "Simulate with a labeled test set?");
		
		if(args.length < 4){

			// automatically generate the help statement

			HelpFormatter formatter = new HelpFormatter();

			formatter.printHelp( "TREC", options);

			}

			parseOptions(options, args);

	}
	
	void parseOptions(Options options,String[] args){

		CommandLineParser parser = new DefaultParser();

		try {

			CommandLine cmd = parser.parse(options, args);

			file = cmd.getOptionValue("file");

			lableInfo = cmd.getOptionValue("lableinfo");

			source = cmd.getOptionValue("source"); // to get all files under the directory

			experimental = cmd.hasOption("experimental");


		} catch (ParseException e) {

			e.printStackTrace();
		}
	}
}
