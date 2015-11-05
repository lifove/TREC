package ca.uwaterloo.tool.trec;

import hk.ust.cse.ipam.utils.ArrayListUtil;
import hk.ust.cse.ipam.utils.HashMapUtil;

import java.util.ArrayList;
import java.util.HashMap;

import org.rosuda.REngine.REXPMismatchException;
import org.rosuda.REngine.REngineException;
import org.rosuda.REngine.RList;
import org.rosuda.REngine.Rserve.RConnection;
import org.rosuda.REngine.Rserve.RserveException;

import weka.core.Instances;

public class DatasetAnalyzer {
	
	Instances tarInsts,tarTrain;
	HashMap<String,Instances> sourceDatasets;
	HashMap<String,Double> similarityScores;
	
	String[] buggyLabels = {"buggy","TRUE"};
	
	DatasetAnalyzer(Instances tarInstances,HashMap<String,Instances> sources){
		
		tarInsts = tarInstances;
		sourceDatasets = sources;
		
	}

	void analyze(){
		// compute similarity scores
		similarityScores = getSimilarityScore(tarInsts,sourceDatasets);
		
		// order similarity scores or get top n datasets		
	}
	
	HashMap<String,Double> getSimilarityScore(Instances tarInstances,HashMap<String,Instances> sources){
		HashMap<String,Double> scores = new HashMap<String,Double>();
		
		for(String srcFile:sources.keySet()){
				
			if(isSameAttributes(tarInstances,sources.get(srcFile))){
				Double score = computeSimilarityOfHomogeneousDatasets(tarInstances,sources.get(srcFile));
				scores.put(srcFile, score);
			}
			
			// conduct Heterogeneous computation for all datasets including the datasets with the same attributes
			Double score = computeSimilarityOfHeterogeneousDatasets(tarInstances,sources.get(srcFile));
			scores.put(srcFile + "(H)", score);
		}
		
		return scores;
	}

	private Double computeSimilarityOfHomogeneousDatasets(Instances tarInstances,
			Instances srcInstances) {

		Double sumPValues = 0.0;
		
		// use KS-test
		int count = 0;
		for(int attrIdx = 0; attrIdx < tarInstances.numAttributes();attrIdx++){
		
			// skip the last (class) attribute
			if(attrIdx==tarInstances.classIndex())
				continue;
			
			double[] tarAttrValues = tarInstances.attributeToDoubleArray(attrIdx);
			double[] srcAttrValues = srcInstances.attributeToDoubleArray(attrIdx);
			
			double pValue= getKSPvalueFromR(srcAttrValues, tarAttrValues);
			
			if(pValue>0.05){
				sumPValues+=pValue;
				count++;
			}
			
		}
		
		if(count==0)
			return -1.0;

		return sumPValues/count++;
	}
	
	private Double computeSimilarityOfHeterogeneousDatasets(Instances tarInstances,
			Instances srcInstances) {

		HashMap<String,Double> matchedPValues = new HashMap<String,Double>(); // src-tgt , pValue

		// use KS-test, compute all matching scores
		for(int tarAttrIdx = 0; tarAttrIdx < tarInstances.numAttributes();tarAttrIdx++){
		
			// skip the last (class) attribute
			if(tarAttrIdx==tarInstances.classIndex())
				continue;
			
			for(int srcAttrIdx = 0; srcAttrIdx < srcInstances.numAttributes();srcAttrIdx++){
				
				// skip the last (class) attribute
				if(srcAttrIdx==srcInstances.classIndex())
					continue;
				
				double[] tarAttrValues = tarInstances.attributeToDoubleArray(tarAttrIdx);
				double[] srcAttrValues = srcInstances.attributeToDoubleArray(tarAttrIdx);
				
				double pValue= getKSPvalueFromR(srcAttrValues, tarAttrValues);
				
				matchedPValues.put(srcAttrIdx + "-" + tarAttrIdx,pValue);
			}
		}

		// find the best matching by a simple greedy approach
		// sort HashMap
		matchedPValues = (HashMap<String, Double>) HashMapUtil.sortByValue(matchedPValues);
		
		ArrayList<Integer> matchedSrcAttrIdx = new ArrayList<Integer>();
		ArrayList<Integer> matchedTarAttrIdx = new ArrayList<Integer>();
		ArrayList<Double> pValues = new ArrayList<Double>(); 
		for(String key:matchedPValues.keySet()){
			String[] srcTarAttrIdices = key.split("-");
			int srcAttrIdx = Integer.parseInt(srcTarAttrIdices[0]);
			int tarAttrIdx = Integer.parseInt(srcTarAttrIdices[1]);
			
			if(matchedSrcAttrIdx.contains(srcAttrIdx) || matchedTarAttrIdx.contains(tarAttrIdx))
				continue;
			
			if(matchedPValues.get(key)>0.05){
				matchedSrcAttrIdx.add(srcAttrIdx);
				matchedTarAttrIdx.add(tarAttrIdx);
				pValues.add(matchedPValues.get(key));
			}
		}
		
		if(pValues.size()==0)
			return -1.0;
		
		return ArrayListUtil.getAverage(pValues);
	}

	// to user Rserve only once, make this method as static.
	// to avoid multiple threads access this method at the same time, synchronized
	RConnection c=null;
	double getKSPvalueFromR(double[] sourceAttrValues,
			double[] targetAttrValues) {
		
		double pValue=0.0;
		try {
			
			// connect once
			if(c==null){
				c = new RConnection();
			}
			
			c.assign("treated", sourceAttrValues);
			c.assign("control", targetAttrValues);
			RList l = c.eval("ks.test(control,treated,exact=TRUE)").asList();
			pValue = l.at("p.value").asDouble();
			
		} catch (RserveException e) {
			e.printStackTrace();
			System.exit(0);
		} catch (REXPMismatchException e) {
			e.printStackTrace();
			System.exit(0);
		} catch (REngineException e) {
			e.printStackTrace();
			System.exit(0);
		}
		return pValue;
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
