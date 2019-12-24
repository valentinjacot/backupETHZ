package patterns.inheritance;

import java.io.ByteArrayOutputStream;
import java.io.IOException;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import patterns.inheritance.StringOutputStream.StringOutputStreamImpl;

@RunWith(Parameterized.class)
public class StringOutputStreamTest extends BaseOutputStreamTest {
	private StringOutputStreamImpl impl;
	private CountedOutputStream cos;
	private ByteArrayOutputStream os;

	public StringOutputStreamTest(StringOutputStreamImpl impl) {
		this.impl = impl;
	}

	@Before
	public void setUp() {
		os = new ByteArrayOutputStream();
		cos = new CountedOutputStream(os);
		cos.setStringOutputStreamImpl(impl);
	}

	@Test
	public void testWrite1() throws IOException {
		cos.write("Hello");
		os.close();
		byte[] buf = os.toByteArray();
		Assert.assertArrayEquals("error in write(String)", new byte[]{ 'H', 'e', 'l', 'l', 'o' }, buf);
	}

	@Test
	public void testWrite2() throws IOException {
		char[] str = { 'H', 'e', 'l', 'l', 'o' };
		for (char ch : str) {
			cos.write(ch);
		}
		os.close();
		byte[] buf = os.toByteArray();
		Assert.assertArrayEquals("error in write(char)", new byte[]{ 'H', 'e', 'l', 'l', 'o' }, buf);
	}

}

