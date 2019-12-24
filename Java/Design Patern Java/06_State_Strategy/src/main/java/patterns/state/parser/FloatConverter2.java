package patterns.state.parser;

import java.io.BufferedReader;
import java.io.InputStreamReader;

public class FloatConverter2 {
	
	public static void main(String[] args) throws Exception {
		BufferedReader r = new BufferedReader(new InputStreamReader(System.in));
		String s = r.readLine();
		while(s != null && s.length() > 0) {
			try {double d = parseFloat(s); System.out.println(d); }
			catch(IllegalArgumentException e) { System.out.println("Illegal Format"); }
			s = r.readLine();
		}
	}

	static interface State {
		default State handleDigit(FloatData data, int val) { return ERROR; }
		default State handleE(FloatData data) { return ERROR; }
		default State handleDot(FloatData data) { return ERROR; }
		default State handlePlus(FloatData data) { return ERROR; }
		default State handleMinus(FloatData data) { return ERROR; }
	}
	
	static class S0 implements State {
		@Override public State handleDigit(FloatData data, int value) { data.m = value; return S1; }
		@Override public State handleDot(FloatData data) { return S2; }
	}
	
	static class S1 implements State {
		@Override public State handleDigit(FloatData data, int value) { data.m = 10*data.m + value; return S1; }
		@Override public State handleDot(FloatData data) { return S3; }
		@Override public State handleE(FloatData data) { return S4; }
	}
	
	static class S2 implements State {
		@Override public State handleDigit(FloatData data, int value) { data.m = data.m + value/data.quo; data.quo = 10*data.quo; return S3; }
	}
	
	static class S3 implements State {
		@Override public State handleDigit(FloatData data, int value) { data.m = data.m + value/data.quo; data.quo = 10*data.quo; return S3; }
		@Override public State handleE(FloatData data) { return S4; }
	}
	
	static class S4 implements State {
		@Override public State handlePlus(FloatData data) { return S5; }
		@Override public State handleMinus(FloatData data) { data.exp_sign = -1; return S5; }
		@Override public State handleDigit(FloatData data, int value) { data.exp = value; return S6; }
	}
	
	static class S5 implements State {
		@Override public State handleDigit(FloatData data, int value) { data.exp = value; return S6; }
	}
	
	static class S6 implements State {
		@Override public State handleDigit(FloatData data, int value) { data.exp = 10*data.exp + value; return S6; }
	}
	
	static class ErrorState implements State {}

	private static final State ERROR = new ErrorState();
	private static final State S0 = new S0();
	private static final State S1 = new S1();
	private static final State S2 = new S2();
	private static final State S3 = new S3();
	private static final State S4 = new S4();
	private static final State S5 = new S5();
	private static final State S6 = new S6();
	
	static class FloatData {
		double m = 0, quo=10;
		int exp = 0, exp_sign = 1;
		double getValue() { return m * Math.pow(10, exp_sign * exp); }
	}

	private static double parseFloat(String str) {
		State s = S0;
		FloatData data = new FloatData();
		int pos = 0;
		while(s != ERROR && pos < str.length()) {
			char ch = str.charAt(pos++);
			if(Character.isDigit(ch)) s = s.handleDigit(data, Character.getNumericValue(ch));
			else if(ch == '.') s = s.handleDot(data);
			else if(ch == '+') s = s.handlePlus(data);
			else if(ch == '-') s = s.handleMinus(data);
			else if(ch == 'E') s = s.handleE(data);
			else if(ch == 'e') s = s.handleE(data);
			else s = ERROR;
		}		
		if (s == S3 || s == S6) {
			return data.getValue();
		} else {
			throw new IllegalArgumentException();
		}
	}
}
