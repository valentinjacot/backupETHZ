package patterns.inheritance;

import java.io.*;

public class StringOutputStream extends FilterOutputStream {
	public StringOutputStream(OutputStream s) { super(s); }

	public void write(char ch) throws IOException {
		impl.write(this, ch);
	}

	public void write(String str) throws IOException {
		impl.write(this, str);
	}

	// implementation of the functionality is plugged in using DI
	private StringOutputStreamImpl impl;
	protected void setStringOutputStreamImpl(StringOutputStreamImpl impl) { this.impl = impl; }
	
	protected static interface StringOutputStreamImpl {
		void write(StringOutputStream sos, char ch) throws IOException;
		void write(StringOutputStream sos, String str) throws IOException;
	}
}	
