package ca.uwaterloo.tool.trec;

import java.io.File;
import java.util.HashMap;
import java.util.Random;

import hk.ust.cse.ipam.utils.HashMapUtil;
import hk.ust.cse.ipam.utils.WekaUtils;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;

import weka.core.Instances;

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

	private void run(String[] args) {
		
		processOptions(args);
		
		// load a test set
		Instances instances = WekaUtils.loadArff(file);
		
		// load a pool of sources
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
				
		for(String srcFile:scores.keySet()){
			System.out.println(srcFile + " similarity=" + scores.get(srcFile));
		}
		
		// compute precision recall curve and AUCEC
		
		
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
