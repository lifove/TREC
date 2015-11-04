package ca.uwaterloo.tool.trec;

import java.util.HashMap;

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

	private void run(String[] args) {
		
		processOptions(args);
		
		// load a test set
		Instances instances = WekaUtils.loadArff(file);
		
		// load a pool of sources
		HashMap<String,Instances> sources = WekaUtils.loadArffs(source);
		
		// preprocessing
		instances = preprocessing(instances);
		
		// identify the best sources
		
		
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
	
	String file;
	String lableInfo;
	String source;
	String experimental;
	
	void parseOptions(Options options,String[] args){

		CommandLineParser parser = new DefaultParser();

		try {

			CommandLine cmd = parser.parse(options, args);

			file = cmd.getOptionValue("file");

			lableInfo = cmd.getOptionValue("lableinfo");

			source = cmd.getOptionValue("source"); // to get all files under the directory

			experimental = cmd.getOptionValue("experimental");


		} catch (ParseException e) {

			e.printStackTrace();
		}
	}
}
