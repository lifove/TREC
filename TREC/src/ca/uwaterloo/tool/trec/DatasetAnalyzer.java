package ca.uwaterloo.tool.trec;

import java.util.HashMap;

import weka.core.Instances;

public class DatasetAnalyzer {
	
	Instances tarInsts;
	HashMap<String,Instances> sourceDatasets;
	HashMap<String,Double> similarityScores;
	
	String[] buggyLabels = {"buggy","TRUE"};
	
	DatasetAnalyzer(Instances tarInstances,HashMap<String,Instances> sources){
		
		tarInsts = tarInstances;
		sourceDatasets = sources;
		
		// compute similarity scores
		similarityScores = getSimilarityScore(tarInsts,sourceDatasets);
		
		// order similarity scores or get top n datasets
		
	}
	
	HashMap<String,Double> getSimilarityScore(Instances tarInstances,HashMap<String,Instances> sources){
		HashMap<String,Double> scores = new HashMap<String,Double>();
		
		for(String srcFile:sources.keySet()){
			if(isSameAttributes(tarInstances,sources.get(srcFile))){
				computeSimilarityOfHomogeneousDatasets(tarInstances,sources.get(srcFile));
			}
		}
		
		return scores;
	}

	private void computeSimilarityOfHomogeneousDatasets(Instances tarInstances,
			Instances srcInstances) {
		
		// use KS-test
		for(int attrIdx = 0; attrIdx < tarInstances.numAttributes();attrIdx++){
		
			
			
		}
	}

	private boolean isSameAttributes(Instances tarInstances,
			Instances srcInstances) {
		
		if(tarInstances.numAttributes()!=srcInstances.numAttributes())
			return false;
		
		// check attribute names except for class attribute name
		for(int attrIdx = 0; attrIdx < tarInstances.numAttributes()-1 ; attrIdx++){
			if(!tarInstances.attribute(attrIdx).name().equals(srcInstances.attribute(attrIdx).name()))
				return false;
		}
		
		return true;
	}
}
