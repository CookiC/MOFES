package weka.attributeSelection;

import weka.classifiers.evaluation.StandardEvaluationMetric;

import java.nio.DoubleBuffer;
import java.util.ArrayList;
import java.util.List;

import weka.classifiers.evaluation.AbstractEvaluationMetric;
import weka.core.Instance;
import weka.core.Utils;

public class RegressionEval extends AbstractEvaluationMetric implements StandardEvaluationMetric{
	private static final long serialVersionUID = -1058021221696083139L;
	
	private static double MRE(double Eact,double Eest){
		return Math.abs(Eact-Eest)/Eact;
	}
	
	private double m_SumMRE;
	private int m_SumPredR;
	private ArrayList<Double> m_MREs=new ArrayList<Double>();
	private List<String> m_StatisticNames=new ArrayList<String>();
	
	public RegressionEval(){
		m_StatisticNames.add("MMRE");
		m_StatisticNames.add("FN");
		m_StatisticNames.add("MrMRE");
		m_StatisticNames.add("predR");
	}
	
	@Override
	public boolean appliesToNumericClass(){
		return true;
	}
	
	@Override
	public boolean appliesToNominalClass(){
		return false;
	}
	
	@Override
	public String getMetricDescription(){
		return  "Author:CookiC\n"+
				"Statistic:\n"+
				"    MMRE:Mean magnitude of relative error.\n"+
				"    FN:Feature number.\n"+
				"    MdMRE:Mean magnitude of relative error\n"+
				"    predR:The ratio of MRE(prediction) under 0.25";
	}
	
	@Override
	public double getStatistic(String statName){
		int size=m_MREs.size();
		switch(statName){
			case "MMRE":
				return m_SumMRE/m_baseEvaluation.numInstances();
			case "FN":
				return m_baseEvaluation.getHeader().numAttributes();
			case "MdMRE":
				double[] MREs=new double[size];
				for(int i=0;i<size;++i)
					MREs[i]=m_MREs.get(i);
				if(size%2==1)
					return Utils.kthSmallestValue(MREs, size/2+1);
				else
					return (Utils.kthSmallestValue(MREs, size/2)+Utils.kthSmallestValue(MREs, size/2+1))/2;
			case "predR":
				return (double)m_SumPredR/size;
			default:
				return 0;
		}
	}
	
	@Override
	public String getMetricName(){
		return "RegressionEval";
	};
	
	public double[] getMREs(){
		double[] MREs=new double[m_MREs.size()];
		for(int i=0;i<m_MREs.size();++i)
			MREs[i]=m_MREs.get(i);
		return MREs;
	}
	
	@Override
	public List<String> getStatisticNames(){
		return m_StatisticNames;
	}
	
	@Override
	public boolean statisticIsMaximisable(String statName) {
		switch(statName){
			case "MMRE":
			case "FN":
			case "MdMRE":
				return false;
			case "predR":
			default:
				return true;
		}
	}
	
	@Override
	public String toSummaryString() {
		return null;
	}

	@Override
	public void updateStatsForClassifier(double[] pred, Instance inst) throws Exception {
		if(inst.classAttribute().isNominal()){
		}
		else
			throw new Exception("Can't solve numeric classes!");
	}

	@Override
	public void updateStatsForPredictor(double pred, Instance inst) throws Exception {
		if(inst.classAttribute().isNumeric()){
			double mre=MRE(pred,inst.classValue());
			m_MREs.add(mre);
			m_SumMRE+=mre;
			if(mre<=0.25)
				++m_SumPredR;
		}
		else
			throw new Exception("Can't solve nominal classes!");
	}
}
