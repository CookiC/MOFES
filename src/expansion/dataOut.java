package expansion;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVSaver;

public class dataOut {
	
	private final String root="results";
	
	private String mainDir;
	
	private String subDir="";
	
	public dataOut(String name){
		mainDir=root+"/"+name;
		File file=new File(mainDir);
		file.mkdir();
		mainDir+="/";
	}
	
	public void outCSV(Instances insts) throws IOException{
		File outFile=new File(mainDir+subDir+insts.relationName()+".csv");
		CSVSaver csvSaver=new CSVSaver();
		csvSaver.setFile(outFile);
		csvSaver.setInstances(insts);
		csvSaver.writeBatch();
	}
	
	public void outArff(Instances insts) throws IOException{
		File outFile=new File(mainDir+subDir+insts.relationName()+".arff");
		ArffSaver arffSaver=new ArffSaver();
		arffSaver.setFile(outFile);
		arffSaver.setInstances(insts);
		arffSaver.writeBatch();
	}
	
	public void outCSV(String name,String[] attrNames,int[][] a) throws IOException{
		int i,j;
		FileWriter outFile=new FileWriter(mainDir+subDir+name+".csv");
		for(i=0;i<attrNames.length;++i){
			if(i>0)
				outFile.write(',');
			outFile.write(attrNames[i]);
		}
		outFile.write('\n');
		for(i=0;i<a.length;++i){
			for(j=0;j<a[i].length;++j){
				if(j>0)
					outFile.write(',');
				outFile.write(a[i][j]);
			}
			outFile.write('\n');
		}
		outFile.close();
	}
	
	public void outCSV(String name,String[] attrNames,double[][] a) throws IOException{
		int i,j;
		FileWriter outFile=new FileWriter(mainDir+subDir+name+".csv");
		for(i=0;i<attrNames.length;++i){
			if(i>0)
				outFile.write(',');
			outFile.write(attrNames[i]);
		}
		outFile.write('\n');
		for(i=0;i<a.length;++i){
			for(j=0;j<a[i].length;++j){
				if(j>0)
					outFile.write(',');
				outFile.write(String.format("%.5f",a[i][j]));
			}
			outFile.write('\n');
		}
		outFile.close();
	}
	
	public void outCSV(Recorder record) throws IOException{
		int i,j;
		String[] colNames=record.getColNames();
		ArrayList<Double[]> data=record;
		FileWriter outFile=new FileWriter(mainDir+subDir+record.getName()+".csv");
		for(i=0;i<colNames.length;++i){
			if(i>0)
				outFile.write(',');
			outFile.write(colNames[i]);
		}
		outFile.write('\n');
		for(i=0;i<data.size();++i){
			for(j=0;j<data.get(i).length;++j){
				if(j>0)
					outFile.write(',');
				outFile.write(String.format("%.5f",data.get(i)[j].doubleValue()));
			}
			outFile.write('\n');
		}
		outFile.close();
	}
	
	public void setSubDir(String name){
		File file=new File(mainDir+name);
		file.mkdir();
		subDir=name+"/";
	}
}
