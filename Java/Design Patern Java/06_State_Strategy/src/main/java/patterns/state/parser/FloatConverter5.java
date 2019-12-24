package patterns.state.parser;
import java.io.BufferedReader;
import java.io.InputStreamReader;

public class FloatConverter5 {

	private abstract static class State {
		final String str;
		final double m, quo;
		final int exp, exp_sign;

		State(String str, double m, double quo, int exp, int exp_sign) {
			this.str = str;
			this.m = m;
			this.quo = quo;
			this.exp = exp;
			this.exp_sign = exp_sign;
		}

		abstract State handle();
	}

	private final static class S0 extends State {

		S0(String str, double m, double quo, int exp, int exp_sign) {
			super(str, m, quo, exp, exp_sign);
		}

		@Override
		State handle() {
			Character ch = str.charAt(0);
			if (isDigit(ch)) return new S1(str.substring(1), getNumericValue(ch), quo, exp, exp_sign);
			else if (ch == '.') return new S2(str.substring(1), m, quo, exp, exp_sign);
			else return new Error(str.substring(1), m, quo, exp, exp_sign);
		}
	}

	private final static class S1 extends State {

		S1(String str, double m, double quo, int exp, int exp_sign) {
			super(str, m, quo, exp, exp_sign);
		}

		@Override
		State handle() {
			Character ch = str.charAt(0);
			if (isDigit(ch)) return new S1(str.substring(1), 10 * m + getNumericValue(ch), quo, exp, exp_sign);
			else if (ch == '.') return new S3(str.substring(1), m, quo, exp, exp_sign);
			else if (ch == 'e') return new S4(str.substring(1), m, quo, exp, exp_sign);
			else if (ch == 'E') return new S4(str.substring(1), m, quo, exp, exp_sign);
			else return new Error(str, m, quo, exp, exp_sign);
		}
	}

	private final static class S2 extends State {

		S2(String str, double m, double quo, int exp, int exp_sign) {
			super(str, m, quo, exp, exp_sign);
		}

		@Override
		State handle() {
			Character ch = str.charAt(0);
			if (isDigit(ch)) return new S3(str.substring(1), m + getNumericValue(ch) / quo, quo * 10, exp, exp_sign);
			else return new Error(str, m, quo, exp, exp_sign);
		}
	}

	private final static class S3 extends State {

		S3(String str, double m, double quo, int exp, int exp_sign) {
			super(str, m, quo, exp, exp_sign);
		}

		@Override
		State handle() {
			Character ch = str.charAt(0);
			if (isDigit(ch)) return new S3(str.substring(1), m + getNumericValue(ch) / quo, quo * 10, exp, exp_sign);
			else if (ch == 'e') return new S4(str.substring(1), m, quo, exp, exp_sign);
			else if (ch == 'E') return new S4(str.substring(1), m, quo, exp, exp_sign);
			else return new Error(str, m, quo, exp, exp_sign);
		}
	}

	private final static class S4 extends State {

		S4(String str, double m, double quo, int exp, int exp_sign) {
			super(str, m, quo, exp, exp_sign);
		}

		@Override
		State handle() {
			Character ch = str.charAt(0);
			if (ch == '+') return new S5(str.substring(1), m, quo, exp, exp_sign);
			else if (ch == '-') return new S5(str.substring(1), m, quo, exp, -1);
			else if (isDigit(ch)) return new S6(str.substring(1), m, quo, getNumericValue(ch), exp_sign);
			else return new Error(str, m, quo, exp, exp_sign);
		}
	}

	private final static class S5 extends State {

		S5(String str, double m, double quo, int exp, int exp_sign) {
			super(str, m, quo, exp, exp_sign);
		}

		@Override
		State handle() {
			Character ch = str.charAt(0);
			if (isDigit(ch)) return new S6(str.substring(1), m, quo, getNumericValue(ch), exp_sign);
			else return new Error(str, m, quo, exp, exp_sign);
		}
	}

	private final static class S6 extends State {

		S6(String str, double m, double quo, int exp, int exp_sign) {
			super(str, m, quo, exp, exp_sign);
		}

		@Override
		State handle() {
			Character ch = str.charAt(0);
			if (isDigit(ch)) return new S6(str.substring(1), m, quo, 10 * exp + getNumericValue(ch), exp_sign);
			else return new Error(str, m, quo, exp, exp_sign);
		}
	}

	private final static class Error extends State {

		Error(String str, double m, double quo, int exp, int exp_sign) {
			super(str, m, quo, exp, exp_sign);
		}

		@Override
		State handle() {
			return this;
		}
	}

	public static void main(String[] args) throws Exception {
		BufferedReader r = new BufferedReader(new InputStreamReader(System.in));
		String s = r.readLine();
		while (s != null && s.length() > 0) {
			try {
				double d = parseFloat(s);
				System.out.println(d);
			}
			catch (IllegalArgumentException e) {
				System.out.println("Illegal Format");
			}
			s = r.readLine();
		}
	}

	private static boolean isDigit(char ch) {
		return Character.isDigit(ch);
	}

	private static int getNumericValue(char ch) {
		return Character.getNumericValue(ch);
	}

	private static double parseFloat(String str) {
		State res = new S0(str, 0, 10, 0, 1).handle();
		while (!res.str.isEmpty() && !(res instanceof Error))
			res = res.handle();
		if (res instanceof S3 || res instanceof S6) return res.m * Math.pow(10, res.exp_sign * res.exp);
		else throw new IllegalArgumentException();
	}
}
