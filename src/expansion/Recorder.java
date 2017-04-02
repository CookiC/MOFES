package expansion;

import java.nio.DoubleBuffer;
import java.util.ArrayList;
import java.util.Comparator;
import org.netlib.util.intW;

public class Recorder extends ArrayList<Double[]>{
	private String name;
	
	private String[] colNames;
	
	private int n;
	
	private int m;
	
	public Recorder(String rName,String[] rColNames){
		name=rName;
		colNames=rColNames;
		m=rColNames.length;
	}
	
	public Recorder(Recorder recorder){
		name=recorder.getName();
		colNames=recorder.getColNames();
		m=colNames.length;
	}
	
	public int nrow(){
		return n;
	}
	
	public int ncol(){
		return m;
	}
	
	public void add(double[] row){
		Double[] r=new Double[row.length];
		for(int i=0;i<r.length;++i)
			r[i]=row[i];
		add(r);
	}
	
	
	public String getName(){
		return name;
	}
	
	public String[] getColNames(){
		return colNames;
	}
	
	public void removeLast(){
		remove(size());
	}
	
	public Double[] last(){
		return get(size());
	}
	
	public boolean empty(){
		return isEmpty();
	}
	
	public void delete(int index){
		remove(index);
	}
	
	public void add(Recorder recorder){
		for(Double[] m:recorder)
			add(m);
	}
	
	public Recorder copy(){
		Recorder copy=new Recorder(this);
		for(Double[] m:this)
			copy.add(m);
		return copy;
	}
	
	public void meanCurve(Recorder recorder,Comparator<Double[]> RecordCmp){
		Recorder copy=recorder.copy();
		copy.sort(RecordCmp);
		int L,R;
		double sum;
		Double[] t;
		clear();
		for(L=0,R=0;R<copy.size();L=R){
			sum=0;
			while(R<copy.size()&&copy.get(L)[0].intValue()==copy.get(R)[0].intValue()){
				sum+=copy.get(R)[1];
				++R;
			}
			t=new Double[2];
			t[0]=copy.get(L)[0];
			t[1]=sum/(R-L);
			add(t);
		}
	}
	
	public void paretoCurve(Recorder recorder,Comparator<Double[]> RecordCmp){
		Recorder copy=recorder.copy();
		clear();
		for(Double[] m:copy)
			add(m);
		sort(RecordCmp);
		for(int i=0;i<size()-1;++i)
			while(i<size()-1&&get(i)[1]>=get(i+1)[1])
				delete(i+1);
	}
	
	public String toString(){
		String str;
		str=name+'\n';
		for(String m:colNames)
			str+=m+" ";
		str+='\n';
		for(Double[] m:this){
			for(Double n:m)
				str+=String.format("%.2f",n)+" ";
			str+='\n';
		}
		return str;
	}
}
