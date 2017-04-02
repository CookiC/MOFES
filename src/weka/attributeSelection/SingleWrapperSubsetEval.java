package weka.attributeSelection;

import java.util.BitSet;
import java.util.Random;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.AbstractEvaluationMetric;
import weka.classifiers.evaluation.InformationRetrievalEvaluationMetric;
import weka.core.Instances;
import weka.core.SelectedTag;

public class SingleWrapperSubsetEval extends WrapperSubsetEval{
	private static final long serialVersionUID = 798667082906264638L;

	@Override
	public double evaluateSubset(BitSet subset) throws Exception {
		double evalMetric = 0;
	    BitSet subsetCopy=(BitSet)subset.clone();
	    subsetCopy.set(m_classIndex);
	    Instances trainCopy = new Instances(m_trainInstances.attributeFilter(subsetCopy));

	    AbstractEvaluationMetric pluginMetric = null;
	    String statName = null;
	    String metricName = null;
		
	    m_Evaluation = new Evaluation(trainCopy);
	    Classifier classifierCopy=AbstractClassifier.makeCopy(m_BaseClassifier);
	    classifierCopy.buildClassifier(trainCopy);
	    m_Evaluation.evaluateModel(classifierCopy, trainCopy);
	    switch (m_evaluationMeasure.getID()) {
	    	case EVAL_DEFAULT:
	    		evalMetric = m_Evaluation.errorRate();break;
	    	case EVAL_ACCURACY:
	    		evalMetric = m_Evaluation.errorRate();break;
	    	case EVAL_RMSE:
	    		evalMetric = m_Evaluation.rootMeanSquaredError();break;
	    	case EVAL_MAE:
	    		evalMetric = m_Evaluation.meanAbsoluteError();break;
	    	case EVAL_FMEASURE:
	    		if (m_IRClassVal < 0)
	    			evalMetric = m_Evaluation.weightedFMeasure();
	    		else
	    			evalMetric = m_Evaluation.fMeasure(m_IRClassVal);
	    		break;
	    	case EVAL_AUC:
	    		if (m_IRClassVal < 0)
	    			evalMetric = m_Evaluation.weightedAreaUnderROC();
	    		else
	    			evalMetric = m_Evaluation.areaUnderROC(m_IRClassVal);
	    		break;
	    	case EVAL_AUPRC:
	    		if (m_IRClassVal < 0)
	    			evalMetric = m_Evaluation.weightedAreaUnderPRC();
	    		else
	    			evalMetric = m_Evaluation.areaUnderPRC(m_IRClassVal);
	    		break;
	    	case EVAL_CORRELATION:
	    		evalMetric = m_Evaluation.correlationCoefficient();break;
	    	default:
	    		if (m_evaluationMeasure.getID() >= EVAL_PLUGIN) {
	    	        metricName = ((PluginTag) m_evaluationMeasure).getMetricName();
	    	        statName = ((PluginTag) m_evaluationMeasure).getStatisticName();
	    	        pluginMetric = m_Evaluation.getPluginMetric(metricName);
	    	        if (pluginMetric == null)
	    	          throw new Exception("Metric  " + metricName + " does not seem to be " + "available");
	    	    }
	    		
	    		if (pluginMetric instanceof InformationRetrievalEvaluationMetric)
	    			if (m_IRClassVal < 0)
	    				evalMetric = ((InformationRetrievalEvaluationMetric) pluginMetric).getClassWeightedAverageStatistic(statName);
	    			else
	    				evalMetric = ((InformationRetrievalEvaluationMetric) pluginMetric).getStatistic(statName, m_IRClassVal);
	    		else
	    			evalMetric = pluginMetric.getStatistic(statName);
	    		break;
	      }

	    m_Evaluation = null;

	    switch (m_evaluationMeasure.getID()) {
	    case EVAL_DEFAULT:
	    case EVAL_ACCURACY:
	    case EVAL_RMSE:
	    case EVAL_MAE:
	      if (m_trainInstances.classAttribute().isNominal()
	        && (m_evaluationMeasure.getID() == EVAL_DEFAULT
	          || m_evaluationMeasure.getID() == EVAL_ACCURACY)) {
	        evalMetric = 1 - evalMetric;
	      } else {
	        evalMetric = -evalMetric; // maximize
	      }
	      break;
	    default:
	      if (pluginMetric != null
	        && !pluginMetric.statisticIsMaximisable(statName)) {
	        evalMetric = -evalMetric; // maximize
	      }
	    }

	    return evalMetric;
	}
	
	public double[] evaluateSubset(BitSet subset,String[] statNames) throws Exception {
	    int i;
	    int numObjective=statNames.length;
	    double[] evalMetric=new double[numObjective];
	    BitSet subsetCopy=(BitSet)subset.clone();
	    subsetCopy.set(m_classIndex);
	    Instances trainCopy = new Instances(m_trainInstances.attributeFilter(subsetCopy));
	    SelectedTag[] statTags=new SelectedTag[numObjective];
	    for(i=0;i<numObjective;++i)
	    	statTags[i]=new SelectedTag(statNames[i], TAGS_EVALUATION);
	  
	    AbstractEvaluationMetric pluginMetric = null;
	    String statName = null;
	    String metricName = null;
	    
		m_Evaluation = new Evaluation(trainCopy);
		Classifier classifierCopy=AbstractClassifier.makeCopy(m_BaseClassifier);
	    classifierCopy.buildClassifier(trainCopy);
	    m_Evaluation.evaluateModel(classifierCopy, trainCopy);
		for(i = 0; i < numObjective; ++i){
			setEvaluationMeasure(statTags[i]);
			switch (m_evaluationMeasure.getID()) {
		    	case EVAL_DEFAULT:
		    		evalMetric[i] = m_Evaluation.errorRate();break;
		    	case EVAL_ACCURACY:
		    		evalMetric[i] = m_Evaluation.errorRate();break;
		    	case EVAL_RMSE:
		    		evalMetric[i] = m_Evaluation.rootMeanSquaredError();break;
		    	case EVAL_MAE:
		    		evalMetric[i] = m_Evaluation.meanAbsoluteError();break;
		    	case EVAL_FMEASURE:
		    		if (m_IRClassVal < 0)
		    			evalMetric[i] = m_Evaluation.weightedFMeasure();
		    		else
		    			evalMetric[i] = m_Evaluation.fMeasure(m_IRClassVal);
		    		break;
		    	case EVAL_AUC:
		    		if (m_IRClassVal < 0)
		    			evalMetric[i] = m_Evaluation.weightedAreaUnderROC();
		    		else
		    			evalMetric[i] = m_Evaluation.areaUnderROC(m_IRClassVal);
		    		break;
		    	case EVAL_AUPRC:
		    		if (m_IRClassVal < 0)
		    			evalMetric[i] = m_Evaluation.weightedAreaUnderPRC();
		    		else
		    			evalMetric[i] = m_Evaluation.areaUnderPRC(m_IRClassVal);
		    		break;
		    	case EVAL_CORRELATION:
		    		evalMetric[i] = m_Evaluation.correlationCoefficient();break;
		    	default:
		    		if (m_evaluationMeasure.getID() >= EVAL_PLUGIN) {
		    	        metricName = ((PluginTag) m_evaluationMeasure).getMetricName();
		    	        statName = ((PluginTag) m_evaluationMeasure).getStatisticName();
		    	        pluginMetric = m_Evaluation.getPluginMetric(metricName);
		    	        if (pluginMetric == null)
		    	          throw new Exception("Metric  " + metricName + " does not seem to be " + "available");
		    	    }
		    		
		    		if (pluginMetric instanceof InformationRetrievalEvaluationMetric)
		    			if (m_IRClassVal < 0)
		    				evalMetric[i] = ((InformationRetrievalEvaluationMetric) pluginMetric).getClassWeightedAverageStatistic(statName);
		    			else
		    				evalMetric[i] = ((InformationRetrievalEvaluationMetric) pluginMetric).getStatistic(statName, m_IRClassVal);
		    		else
		    			evalMetric[i] = pluginMetric.getStatistic(statName);
		    		break;
			}
			switch (m_evaluationMeasure.getID()) {
				case EVAL_DEFAULT:
				case EVAL_ACCURACY:
				case EVAL_RMSE:
				case EVAL_MAE:
					if (m_trainInstances.classAttribute().isNominal()
						&& (m_evaluationMeasure.getID() == EVAL_DEFAULT
						|| m_evaluationMeasure.getID() == EVAL_ACCURACY))
						evalMetric[i] = 1 - evalMetric[i];
					else
						evalMetric[i] = -evalMetric[i]; // maximize
					break;
				default:
					if (pluginMetric != null&& !pluginMetric.statisticIsMaximisable(statName))
						evalMetric[i] = -evalMetric[i]; // maximize
			}
		}
		
		m_Evaluation = null;
		return evalMetric;
	}
}
