package expansion;

import java.util.Random;

import jflex.Out;
import weka.attributeSelection.RegressionEval;
import weka.classifiers.evaluation.AbstractEvaluationMetric;
import weka.core.Instances;
import weka.core.PluginManager;

public abstract class experiment {
	protected Instances dataSet;
	
	protected Instances trainSet;
	
	protected Instances testSet;
	
	protected dataIn input;
	
	protected dataOut output;
	
	protected void randomSplit(Instances dataSet, double d,int seed) throws Exception {
		dataSet.randomize(new Random(seed));
		int dataSize = dataSet.numInstances();
		int trainSize = (int)Math.round(dataSize * d);
        int testSize = dataSize - trainSize;
        trainSet = new Instances(dataSet, 0, trainSize);
        testSet = new Instances(dataSet, trainSize, testSize);
	}
	
	public experiment(){
		PluginManager.addPlugin(AbstractEvaluationMetric.class.getName(),"RegressionEval",RegressionEval.class.getName());
	}
}
