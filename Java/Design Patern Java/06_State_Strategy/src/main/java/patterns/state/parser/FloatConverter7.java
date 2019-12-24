package patterns.state.parser;

import java.io.BufferedReader;
import java.io.InputStreamReader;

// This is a recursive functional version. For large strings (e.g. more than 18'000 characters) a StackOverflowError is thrown
// as Java does not support tail recursion.
public class FloatConverter7 {

	@FunctionalInterface
	private interface State {
		double handle(String str, double m, double quo, int exp, int exp_sign);
	}

	private final static State s0 = (String str, double m, double quo, int exp, int exp_sign) -> {
		if (str.isEmpty()) return FloatConverter7.error.handle(str, m, quo, exp, exp_sign);
		else {
			Character ch = str.charAt(0);
			if (isDigit(ch)) return FloatConverter7.s1.handle(str.substring(1), getNumericValue(ch), quo, exp, exp_sign);
			else if (ch == '.') return FloatConverter7.s2.handle(str.substring(1), m, quo, exp, exp_sign);
			else return FloatConverter7.error.handle(str.substring(1), m, quo, exp, exp_sign);
		}
	};
	private final static State s1 = (String str, double m, double quo, int exp, int exp_sign) -> {
		if (str.isEmpty()) return FloatConverter7.error.handle(str, m, quo, exp, exp_sign);
		else {
			Character ch = str.charAt(0);
			if (isDigit(ch))
				return FloatConverter7.s1.handle(str.substring(1), 10 * m + getNumericValue(ch), quo, exp, exp_sign);
			else if (ch == '.') return FloatConverter7.s3.handle(str.substring(1), m, quo, exp, exp_sign);
			else if (ch == 'e') return FloatConverter7.s4.handle(str.substring(1), m, quo, exp, exp_sign);
			else if (ch == 'E') return FloatConverter7.s4.handle(str.substring(1), m, quo, exp, exp_sign);
			else return FloatConverter7.error.handle(str, m, quo, exp, exp_sign);
		}
	};
	private final static State s2 = (String str, double m, double quo, int exp, int exp_sign) -> {
		if (str.isEmpty()) return FloatConverter7.error.handle(str, m, quo, exp, exp_sign);
		else {
			Character ch = str.charAt(0);
			if (isDigit(ch))
				return FloatConverter7.s3.handle(str.substring(1), m + getNumericValue(ch) / quo, quo * 10, exp, exp_sign);
			else return FloatConverter7.error.handle(str, m, quo, exp, exp_sign);
		}
	};
	private final static State s3 = (String str, double m, double quo, int exp, int exp_sign) -> {
		if (str.isEmpty()) return m * Math.pow(10, exp_sign * exp);
		else {
			Character ch = str.charAt(0);
			if (isDigit(ch))
				return FloatConverter7.s3.handle(str.substring(1), m + getNumericValue(ch) / quo, quo * 10, exp, exp_sign);
			else if (ch == 'e') return FloatConverter7.s4.handle(str.substring(1), m, quo, exp, exp_sign);
			else if (ch == 'E') return FloatConverter7.s4.handle(str.substring(1), m, quo, exp, exp_sign);
			else return FloatConverter7.error.handle(str, m, quo, exp, exp_sign);
		}
	};
	private final static State s4 = (String str, double m, double quo, int exp, int exp_sign) -> {
		if (str.isEmpty()) return FloatConverter7.error.handle(str, m, quo, exp, exp_sign);
		else {
			Character ch = str.charAt(0);
			if (ch == '+') return FloatConverter7.s5.handle(str.substring(1), m, quo, exp, exp_sign);
			else if (ch == '-') return FloatConverter7.s5.handle(str.substring(1), m, quo, exp, -1);
			else if (isDigit(ch)) return FloatConverter7.s6.handle(str.substring(1), m, quo, getNumericValue(ch), exp_sign);
			else return FloatConverter7.error.handle(str, m, quo, exp, exp_sign);
		}
	};
	private final static State s5 = (String str, double m, double quo, int exp, int exp_sign) -> {
		if (str.isEmpty()) return FloatConverter7.error.handle(str, m, quo, exp, exp_sign);
		else {
			Character ch = str.charAt(0);
			if (isDigit(ch)) return FloatConverter7.s6.handle(str.substring(1), m, quo, getNumericValue(ch), exp_sign);
			else return FloatConverter7.error.handle(str, m, quo, exp, exp_sign);
		}
	};
	private final static State s6 = (String str, double m, double quo, int exp, int exp_sign) -> {
		if (str.isEmpty()) return m * Math.pow(10, exp_sign * exp);
		else {
			Character ch = str.charAt(0);
			if (isDigit(ch))
				return FloatConverter7.s6.handle(str.substring(1), m, quo, 10 * exp + getNumericValue(ch), exp_sign);
			else return FloatConverter7.error.handle(str, m, quo, exp, exp_sign);
		}
	};
	private final static State error = (String str, double m, double quo, int exp, int exp_sign) -> {
		throw new IllegalArgumentException();
	};

	public static void main(String[] args) throws Exception {
		System.out.println("In: ");
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
		double m = 0, quo = 10;
		int exp = 0, exp_sign = 1;
		return s0.handle(str, m, quo, exp, exp_sign);
	}
}
