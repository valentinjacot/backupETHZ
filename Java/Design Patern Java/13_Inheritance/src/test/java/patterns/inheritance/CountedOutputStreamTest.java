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
public class CountedOutputStreamTest extends BaseOutputStreamTest {
	private StringOutputStreamImpl impl;
	private CountedOutputStream cos;

	public CountedOutputStreamTest(StringOutputStreamImpl impl) {
		this.impl = impl;
	}

	@Before
	public void setUp() {
		ByteArrayOutputStream os = new ByteArrayOutputStream();
		cos = new CountedOutputStream(os);
		cos.setStringOutputStreamImpl(impl);
	}

	@Test
	public void testCounter1() throws IOException {
		cos.write("Hello");
		Assert.assertEquals(
				"wrong counter result when writing \"Hello\" with method write(String)",
				5, cos.writtenChars());
	}

	@Test
	public void testCounter2() throws IOException {
		char[] str = { 'H', 'e', 'l', 'l', 'o' };
		for (char ch : str) {
			cos.write(ch);
		}
		Assert.assertEquals(
				"wrong counter result when writing \"Hello\" with method write(char)",
				5, cos.writtenChars());
	}

}

