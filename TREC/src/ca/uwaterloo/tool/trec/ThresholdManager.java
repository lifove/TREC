package ca.uwaterloo.tool.trec;

import java.util.ArrayList;

public class ThresholdManager {
	
	final int IDX_TP = 0;
	final int IDX_FP = 1;
	final int IDX_TN = 2;
	final int IDX_FN = 3;
	final int IDX_PRECISON = 4;
	final int IDX_RECALL = 5;
	final int IDX_F1 = 6;
	final int IDX_THD = 7;
	
	ArrayList<String> mResults;
	
	ThresholdManager(ArrayList<String> results){
		mResults = results;
	}
	
	public Measure processResultsToGetBestInfo(int idxLookAt){
		
		double bestValueLookAt = -1.0;
		int TP=0,FP=0,TN=0,FN=0;
		double precision=0.0,recall=0.0,f1=0.0;
		double threshold=0.0;
		double percentileRankOfThreshold=0.0;
		
		for(int i=0;i<mResults.size();i++){
		
			String[] splitLine = mResults.get(i).split(",");
			
			double valueLookAt = Double.parseDouble(splitLine[idxLookAt]);
			
			if(valueLookAt>bestValueLookAt && valueLookAt!=1.0){
				bestValueLookAt = valueLookAt;
				
				TP = (int) Double.parseDouble(splitLine[IDX_TP]);
				FP = (int) Double.parseDouble(splitLine[IDX_FP]);
				TN = (int) Double.parseDouble(splitLine[IDX_TN]);
				FN = (int) Double.parseDouble(splitLine[IDX_FN]);
				precision = Double.parseDouble(splitLine[IDX_PRECISON]);
				recall = Double.parseDouble(splitLine[IDX_RECALL]);
				f1 = Double.parseDouble(splitLine[IDX_F1]);
				threshold = Double.parseDouble(splitLine[IDX_THD]);
				
				percentileRankOfThreshold = (double)(i+1)/mResults.size();
			}
		}
		
		return new Measure(TP,FP,TN,FN,precision,recall,f1,threshold,percentileRankOfThreshold);
	}

	public Measure getResultByThresholdPercentileRank(double percentileRankOfThresholdFromSrc) {
		
		int TP=0,FP=0,TN=0,FN=0;
		double precision=0.0,recall=0.0,f1=0.0;
		double threshold=0.0;
		double percentileRankOfThreshold=0.0;
		
		double position = mResults.size() * percentileRankOfThresholdFromSrc;
		
		int intPosition = (int)Math.round(position)-1;
		
		String[] splitLine = mResults.get(intPosition==-1?0:intPosition).split(",");
		
		TP = (int) Double.parseDouble(splitLine[IDX_TP]);
		FP = (int) Double.parseDouble(splitLine[IDX_FP]);
		TN = (int) Double.parseDouble(splitLine[IDX_TN]);
		FN = (int) Double.parseDouble(splitLine[IDX_FN]);
		precision = Double.parseDouble(splitLine[IDX_PRECISON]);
		recall = Double.parseDouble(splitLine[IDX_RECALL]);
		f1 = Double.parseDouble(splitLine[IDX_F1]);
		threshold = Double.parseDouble(splitLine[IDX_THD]);
		
		percentileRankOfThreshold = (double)((intPosition==-1?0:intPosition)+1)/mResults.size();
		
		return new Measure(TP,FP,TN,FN,precision,recall,f1,threshold,percentileRankOfThreshold);
	}

	public Measure getResultByThreshold(double threshodFromSrc) {
		int TP=0,FP=0,TN=0,FN=0;
		double precision=0.0,recall=0.0,f1=0.0;
		double threshold=0.0;
		double percentileRankOfThreshold=0.0;
		
		int idxResult = 0;
		
		for(int i=0;i<mResults.size();i++){
			String strThreshold = mResults.get(i).split(",")[IDX_THD];
			
			if (Double.parseDouble(strThreshold)>=threshodFromSrc){
				idxResult = i;
				break;
			}
		}
		
		String[] splitLine = mResults.get(idxResult).split(",");
		
		TP = (int) Double.parseDouble(splitLine[IDX_TP]);
		FP = (int) Double.parseDouble(splitLine[IDX_FP]);
		TN = (int) Double.parseDouble(splitLine[IDX_TN]);
		FN = (int) Double.parseDouble(splitLine[IDX_FN]);
		precision = Double.parseDouble(splitLine[IDX_PRECISON]);
		recall = Double.parseDouble(splitLine[IDX_RECALL]);
		f1 = Double.parseDouble(splitLine[IDX_F1]);
		threshold = Double.parseDouble(splitLine[IDX_THD]);
		
		percentileRankOfThreshold = (double)(idxResult+1)/mResults.size();
		
		return new Measure(TP,FP,TN,FN,precision,recall,f1,threshold,percentileRankOfThreshold);
	}
}

class Measure{

	int mTP,mFP,mTN,mFN;
	double mPrecision,mRecall,mF1;
	double mThreshold;
	double mPercentileRankOfThreshold;
	
	Measure(int TP,int FP,int TN,int FN,double precision,double recall,double f1,double threshold,double percentileRankOfThreshold){
		mTP = TP;
		mFP = FP;
		mTN = TN;
		mFN = FN;
		mPrecision = precision;
		mRecall = recall;
		mF1 = f1;
		mThreshold = threshold;
		mPercentileRankOfThreshold = percentileRankOfThreshold;
	}
	
	public int getTP() {
		return mTP;
	}

	public int getFP() {
		return mFP;
	}

	public int getTN() {
		return mTN;
	}

	public int getFN() {
		return mFN;
	}

	public double getPrecision() {
		return mPrecision;
	}

	public double getRecall() {
		return mRecall;
	}

	public double getF1() {
		return mF1;
	}

	public double getThreshold() {
		return mThreshold;
	}

	public double getPercentileRankOfThreshold() {
		return mPercentileRankOfThreshold;
	}
}