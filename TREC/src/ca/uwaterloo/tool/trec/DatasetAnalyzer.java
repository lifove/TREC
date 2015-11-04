package ca.uwaterloo.tool.trec;

import java.util.HashMap;

import org.rosuda.REngine.REXPMismatchException;
import org.rosuda.REngine.REngineException;
import org.rosuda.REngine.RList;
import org.rosuda.REngine.Rserve.RConnection;
import org.rosuda.REngine.Rserve.RserveException;

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
				Double score = computeSimilarityOfHomogeneousDatasets(tarInstances,sources.get(srcFile));
				scores.put(srcFile, score);
			}
		}
		
		return scores;
	}

	private Double computeSimilarityOfHomogeneousDatasets(Instances tarInstances,
			Instances srcInstances) {

		Double sumPValues = 0.0;
		
		// use KS-test
		for(int attrIdx = 0; attrIdx < tarInstances.numAttributes();attrIdx++){
		
			// skip the last (class) attribute
			if(attrIdx==tarInstances.classIndex())
				continue;
			
			double[] tarAttrValues = tarInstances.attributeToDoubleArray(attrIdx);
			double[] srcAttrValues = srcInstances.attributeToDoubleArray(attrIdx);
			
			double pValue= getKSPvalueFromR(srcAttrValues, tarAttrValues);
			sumPValues+=pValue;
			
		}

		return sumPValues/(tarInstances.numAttributes()-1);
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
