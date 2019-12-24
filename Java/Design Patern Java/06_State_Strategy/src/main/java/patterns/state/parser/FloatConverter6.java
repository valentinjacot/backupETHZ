package patterns.state.parser;
import java.io.BufferedReader;
import java.io.InputStreamReader;

public class FloatConverter6 {

	private final static class Result {
		final State s;
		final String str;
		final double m, quo;
		final int exp, exp_sign;

		Result(State s, String str, double m, double quo, int exp, int exp_sign) {
			this.s = s;
			this.str = str;
			this.m = m;
			this.quo = quo;
			this.exp = exp;
			this.exp_sign = exp_sign;
		}
	}

	@FunctionalInterface
	private interface State {
		Result handle(String str, double m, double quo, int exp, int exp_sign);
	}

	private final static State s0 = (String str, double m, double quo, int exp, int exp_sign) -> {
		Character ch = str.charAt(0);
		if (isDigit(ch)) return new Result(FloatConverter6.s1, str.substring(1), getNumericValue(ch), quo, exp, exp_sign);
		else if (ch == '.') return new Result(FloatConverter6.s2, str.substring(1), m, quo, exp, exp_sign);
		else return new Result(FloatConverter6.error, str.substring(1), m, quo, exp, exp_sign);
	};
	private final static State s1 = (String str, double m, double quo, int exp, int exp_sign) -> {
		Character ch = str.charAt(0);
		if (isDigit(ch))
			return new Result(FloatConverter6.s1, str.substring(1), 10 * m + getNumericValue(ch), quo, exp, exp_sign);
		else if (ch == '.') return new Result(FloatConverter6.s3, str.substring(1), m, quo, exp, exp_sign);
		else if (ch == 'e') return new Result(FloatConverter6.s4, str.substring(1), m, quo, exp, exp_sign);
		else if (ch == 'E') return new Result(FloatConverter6.s4, str.substring(1), m, quo, exp, exp_sign);
		else return new Result(FloatConverter6.error, str, m, quo, exp, exp_sign);
	};
	private final static State s2 = (String str, double m, double quo, int exp, int exp_sign) -> {
		Character ch = str.charAt(0);
		if (isDigit(ch)) return new Result(FloatConverter6.s3, str.substring(1), m + getNumericValue(ch) / quo, quo * 10, exp, exp_sign);
		else return new Result(FloatConverter6.error, str, m, quo, exp, exp_sign);
	};
	private final static State s3 = (String str, double m, double quo, int exp, int exp_sign) -> {
		Character ch = str.charAt(0);
		if (isDigit(ch)) return new Result(FloatConverter6.s3, str.substring(1), m + getNumericValue(ch) / quo,
				quo * 10, exp, exp_sign);
		else if (ch == 'e') return new Result(FloatConverter6.s4, str.substring(1), m, quo, exp, exp_sign);
		else if (ch == 'E') return new Result(FloatConverter6.s4, str.substring(1), m, quo, exp, exp_sign);
		else return new Result(FloatConverter6.error, str, m, quo, exp, exp_sign);
	};
	private final static State s4 = (String str, double m, double quo, int exp, int exp_sign) -> {
		Character ch = str.charAt(0);
		if (ch == '+') return new Result(FloatConverter6.s5, str.substring(1), m, quo, exp, exp_sign);
		else if (ch == '-') return new Result(FloatConverter6.s5, str.substring(1), m, quo, exp, -1);
		else if (isDigit(ch)) return new Result(FloatConverter6.s6, str.substring(1), m, quo, getNumericValue(ch), exp_sign);
		else return new Result(FloatConverter6.error, str, m, quo, exp, exp_sign);
	};
	private final static State s5 = (String str, double m, double quo, int exp, int exp_sign) -> {
		Character ch = str.charAt(0);
		if (isDigit(ch)) return new Result(FloatConverter6.s6, str.substring(1), m, quo, getNumericValue(ch), exp_sign);
		else return new Result(FloatConverter6.error, str, m, quo, exp, exp_sign);
	};
	private final static State s6 = (String str, double m, double quo, int exp, int exp_sign) -> {
		Character ch = str.charAt(0);
		if (isDigit(ch))
			return new Result(FloatConverter6.s6, str.substring(1), m, quo, 10 * exp + getNumericValue(ch), exp_sign);
		else return new Result(FloatConverter6.error, str, m, quo, exp, exp_sign);
	};
	private final static State error = (String str, double m, double quo, int exp, int exp_sign) -> {
		throw new IllegalArgumentException();
	};

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
		Result res = s0.handle(str, 0, 10, 0, 1);
		while (!res.str.isEmpty())
			res = res.s.handle(res.str, res.m, res.quo, res.exp, res.exp_sign);
		if (res.s == s3 || res.s == s6) return res.m * Math.pow(10, res.exp_sign * res.exp);
		else throw new IllegalArgumentException();
	}
}
