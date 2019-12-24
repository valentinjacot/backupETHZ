package patterns.inheritance;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import org.junit.runners.Parameterized.Parameters;

import patterns.inheritance.StringOutputStream.StringOutputStreamImpl;

public class BaseOutputStreamTest {

	@Parameters
	public static List<Object[]> getParams() {
		Object[][] figs = new Object[][] { 
				{ new StringOutputStreamImpl1() }, 
				{ new StringOutputStreamImpl2() }, 
				{ new StringOutputStreamImpl3() }, 
				{ new StringOutputStreamImpl4() }, 
				{ new StringOutputStreamImpl5() }, 
				{ new StringOutputStreamImpl6() }, 
		};
		return Arrays.asList(figs);
	}

	static class StringOutputStreamImpl1 implements StringOutputStreamImpl {
		@Override
		public void write(StringOutputStream sos, char ch) throws IOException {
			sos.write((int)ch);
		}
		@Override
		public void write(StringOutputStream sos, String s) throws IOException {
			sos.write(s.getBytes());
		}
	}
	static class StringOutputStreamImpl2 implements StringOutputStreamImpl {
		@Override
		public void write(StringOutputStream sos, char ch) throws IOException {
			sos.write((int)ch);
		}
		@Override
		public void write(StringOutputStream sos, String s) throws IOException {
			for(int i=0; i<s.length(); i++) sos.write(s.charAt(i));
		}
	}
	static class StringOutputStreamImpl3 implements StringOutputStreamImpl {
		@Override
		public void write(StringOutputStream sos, char ch) throws IOException {
			sos.write(new String(new char[]{ch}));
		}
		@Override
		public void write(StringOutputStream sos, String s) throws IOException {
			sos.write(s.getBytes());
		}
	}
	static class StringOutputStreamImpl4 implements StringOutputStreamImpl {
		@Override
		public void write(StringOutputStream sos, char ch) throws IOException {
			sos.write((int)ch);
		}
		@Override
		public void write(StringOutputStream sos, String s) throws IOException {
			sos.write(s.charAt(0));
			if(s.length() > 1) sos.write(s.substring(1));
		}
	}
	static class StringOutputStreamImpl5 implements StringOutputStreamImpl {
		@Override
		public void write(StringOutputStream sos, char ch) throws IOException {
			sos.write((int)ch);
		}
		@Override
		public void write(StringOutputStream sos, String s) throws IOException {
			sos.write(s.charAt(0));
			sos.write(s.substring(1).getBytes());
		}
	}
	static class StringOutputStreamImpl6 implements StringOutputStreamImpl {
		@Override
		public void write(StringOutputStream sos, char ch) throws IOException {
			sos.write((int)ch);
		}

		@Override
		public void write(StringOutputStream sos, String s) throws IOException {
			if (s.length() == 1) {
				sos.write(s.charAt(0));
			} else {
				int n1 = s.length() / 2;
				int n2 = s.length() - n1;
				if (n1 > 0) {
					sos.write(s.substring(0, n1));
				}
				if (n2 > 0) {
					sos.write(s.substring(n1, n1 + n2));
				}
			}
		}
	}

}
