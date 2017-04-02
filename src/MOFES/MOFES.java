package MOFES;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.DoubleBuffer;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Random;
import java.util.prefs.PreferenceChangeEvent;

import javax.print.attribute.standard.MediaName;
import javax.swing.text.html.HTML.Tag;

import SEE.kNNRegression;
import expansion.Recorder;
import expansion.dataIn;
import expansion.dataOut;
import weka.attributeSelection.ASSearch;
import weka.attributeSelection.GeneticSearch;
import weka.attributeSelection.GreedyStepwise;
import weka.attributeSelection.NSGAII;
import weka.attributeSelection.RegressionEval;
import weka.attributeSelection.SingleWrapperSubsetEval;
import weka.attributeSelection.WrapperSubsetEval;
import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.AbstractEvaluationMetric;
import weka.classifiers.functions.Logistic;
import weka.classifiers.lazy.IBk;
import weka.core.Attribute;
import weka.core.Instances;
import weka.gui.knowledgeflow.steps.ImageViewerInteractiveView;

public class MOFES extends expansion.experiment{
	private final String dataSetsName="PROMISE";
	
	private final String[] searchNames={"GFS","GBS","SOFS","Full","MOFES"};
	
	private final String[] statNames={"FN","AUC"};
	
	private final int runtimes=10;
	
	private final Random rnd=new Random(13);
	
	private ASSearch[] searchs=new ASSearch[5];
	
	private ArrayList<Attribute> resAttrs=new ArrayList<Attribute>();
	
	private double[][] time;
	
	private double[][] ratio;
	
	private Recorder[] perf;
	
	private class RecordCmp implements Comparator<Double[]>{

		@Override
		public int compare(Double[] a, Double[] b) {
			if(a[0]>b[0])
				return 1;
			if(a[0]<b[0])
				return -1;
			if(a[1]<b[1])
				return 1;
			if(a[1]>b[1])
				return -1;
			return 0;
		}
	}
	
	public MOFES() throws FileNotFoundException, IOException{
		input=new dataIn(dataSetsName);
		output=new dataOut(dataSetsName+"1");
		dataSet=input.getDataSet(0);
		time=new double[input.size()][5];
		ratio=new double[input.size()*runtimes][dataSet.numAttributes()-1];
		GreedyStepwise FW=new GreedyStepwise();
		FW.setSearchBackwards(false);
		GreedyStepwise BW=new GreedyStepwise();
		BW.setSearchBackwards(true);
		GeneticSearch GS=new GeneticSearch();
		GS.setPopulationSize(40);
		GS.setMaxGenerations(30);
		GS.setCrossoverProb(0.2);
		GS.setMutationProb(0.6);
		NSGAII nsgaii=new NSGAII();
		nsgaii.setPopulationSize(40);
		nsgaii.setMaxGenerations(30);
		nsgaii.setCrossoverProb(0.2);
		nsgaii.setMutationProb(0.6);
		searchs[0]=FW;
		searchs[1]=BW;
		searchs[2]=GS;
		searchs[3]=null;
		searchs[4]=nsgaii;
		for(String m:statNames)
			resAttrs.add(new Attribute(m));
		
		perf=new Recorder[6];
		String[] perfNames={"Feature number","AUC"};
		for(int i=0;i<4;++i)
			perf[i]=new Recorder(searchNames[i], perfNames);
		perf[4]=new Recorder("MOFES-A", perfNames);
		perf[5]=new Recorder("MOFES-B", perfNames);
	}
	
	private long runOnce(ASSearch search,double ratio[],Recorder perfs,int seed) throws Exception {
		int i,j;
		long start;
		long time=0;
		IBk classifer=new IBk(7);
		WrapperSubsetEval wrap=new WrapperSubsetEval();
		wrap.setClassifier(classifer);
		wrap.setEvaluationMeasure("auc");
		wrap.setFolds(3);
		wrap.buildEvaluator(trainSet);
		wrap.setSeed(seed);

		int[][] temp;
		start=System.currentTimeMillis();
		if (search instanceof NSGAII){
			temp=((NSGAII)search).search(wrap, trainSet,statNames);
			for(i=0;i<temp.length;++i)
				for(j=0;j<temp[i].length;++j)
					++ratio[temp[i][j]];
			for(i=0;i<dataSet.numAttributes()-1;++i)
				ratio[i]/=temp.length;
		}
		else{
			temp=new int[1][];
			if(search==null){
				temp[0]=new int[dataSet.numAttributes()-1];
				for(i=0,j=0;i<dataSet.numAttributes();++i)
					if(i!=dataSet.classIndex())
						temp[0][j++]=i;
			}
			else
				temp[0]=search.search(wrap, trainSet);
		}
		time+=System.currentTimeMillis()-start;
		
		Recorder perf=new Recorder(perfs);
		for(i=0;i<temp.length;++i){
			int[] attrsIndex=new int[temp[i].length+1];
			for(j=0;j<temp[i].length;++j)
				attrsIndex[j]=temp[i][j];
			
			attrsIndex[j]=dataSet.classIndex();
			Instances trainCopy=trainSet.attributeFilter(attrsIndex);
			Instances testCopy=testSet.attributeFilter(attrsIndex);
			Evaluation eval=new Evaluation(trainCopy);
			classifer.buildClassifier(trainCopy);
			
			Double[] record=new Double[2];
			eval=new Evaluation(trainCopy);
			eval.evaluateModel(classifer, testCopy);
			record[0]=(double)attrsIndex.length-1;
			record[1]=eval.weightedAreaUnderROC();
			perf.add(record);
		}
		
		perf.paretoCurve(perf,new RecordCmp());
		perfs.add(perf);
		return time;
	}
	
	public void run() throws Exception{
		int i,j,k;
		double[][] ratioOut=new double[input.size()*(dataSet.numAttributes()-1)*runtimes][2];
		String[] ratioName={"SelectionRatio","FeatureID"};
		
		for(i=0;i<input.size();++i){
			dataSet=input.getDataSet(i);
			dataSet.setClassIndex("Defective");
			output.setSubDir(dataSet.relationName());
			randomSplit(dataSet, 0.7,1);
			for(j=0;j<runtimes;++j){
				int seed=rnd.nextInt();
				for(k=0;k<searchs.length;++k)
					time[i][k]+=runOnce(searchs[k],ratio[i*runtimes+j],perf[k],seed)/1000.0;
			}
			
			output.setSubDir(dataSet.relationName());
			System.out.println(perf[5]);
			perf[5].meanCurve(perf[4],new RecordCmp());
			System.out.println(perf[5]);
			System.out.println(perf[4]);
			perf[4].paretoCurve(perf[4],new RecordCmp());
			System.out.println(perf[4]);
			for(k=0;k<perf.length;++k){
				output.outCSV(perf[k]);
				perf[k].clear();
			}
		}
		output.setSubDir("");
		output.outCSV("Computational Cost",searchNames,time);
		for(j=0;j<dataSet.numAttributes()-1;++j)
			for(k=0;k<ratio.length;++k){
				ratioOut[j*ratio.length+k][0]=ratio[k][j];
				ratioOut[j*ratio.length+k][1]=j+1;
			}
		output.outCSV("Selection Ratio",ratioName,ratioOut);
	}
	
	public static void main(String[] args)throws Exception{
		MOFES mofes=new MOFES();
		mofes.run();
		//for(weka.core.Tag m:WrapperSubsetEval.TAGS_EVALUATION)
		//	System.out.println(m);
	}
}
