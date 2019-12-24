package patterns.inheritance;

import java.io.*;

public class CountedOutputStream extends StringOutputStream {
	private int c;
	private Integer c1=c;
//	private boolean active = false;

	
	public CountedOutputStream(OutputStream s) {
		super(s);
		c=0;
	}
	
	public int writtenChars() {
		// TODO return number of written characters
		return c;
	}

	@Override
	public void write(char ch) throws IOException {
		// TODO Auto-generated method stub
		int temp=c+1;
		super.write(ch);c=temp;
//		if (! active ) { active=true; super.write(ch); c++; active=false; }
//		else super.write(ch);
	}

	@Override
	public void write(String s) throws IOException {
		// TODO Auto-generated method stub
		int temp=c+s.length();
		super.write(s);c= temp;
//		if (! active ) {active=true; super.write(s); c+=s.length(); active=false;}
//		else super.write(s);
	}

}
