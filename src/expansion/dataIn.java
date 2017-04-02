package expansion;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

import org.netlib.util.intW;

import weka.core.Instances;
import weka.core.converters.CSVLoader;

public class dataIn {
	private final String rootName="data";
	
	private int size;
	private File folder;
	private String folderName;
	private ArrayList<String> fileNames=new ArrayList<String>();
	private String dataSetsName;
	private Instances[] dataSets;
	
	public dataIn(String name) throws FileNotFoundException, IOException{
		reads(name);
	}
	
	private void reads(String name) throws FileNotFoundException, IOException{
		String[] list;
		String type="";
		dataSetsName=name;
		folderName=rootName+"/"+name;
		folder=new File(folderName);
		list=folder.list();
		for(String m:list){
			type=m.substring(m.lastIndexOf('.')+1);
			if(type.equals("arff")||type.equals("csv"))
				fileNames.add(m);
		}
		
		size=fileNames.size();
		dataSets=new Instances[size];
		for(int i=0;i<size;++i){
			String m=fileNames.get(i);
			File frData=new File(folderName+"/"+m);
			if(type.equals("arff"))
				dataSets[i]=new Instances(new FileReader(frData));
			else{
				CSVLoader csvLoader=new CSVLoader();
				csvLoader.setFile(frData);
				dataSets[i]=csvLoader.getDataSet();
			}
		}
	}
	
	public Instances getDataSet(int index){
		return dataSets[index];
	}
	
	public String getSetsName(){
		return dataSetsName;
	}
	
	public int size(){
		return dataSets.length;
	}
	
	public static void main(String[] args)throws Exception{
		dataIn input=new dataIn("PROMISE");
		System.out.println(input.getDataSet(1).relationName());
	}
}
