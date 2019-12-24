package patterns.state.parser;

import java.io.BufferedReader;
import java.io.InputStreamReader;

public class FloatConverter3 {
	
	public static void main(String[] args) throws Exception {
		BufferedReader r = new BufferedReader(new InputStreamReader(System.in));
		String s = r.readLine();
		while(s != null && s.length() > 0){
			try {double d = parseFloat(s); System.out.println(d);}
			catch(IllegalArgumentException e){System.out.println("Illegal Format");}
			s = r.readLine();
		}
	}
	
	enum State {
		S0() {
			@Override public State handleDigit(FloatData data, int value) { data.m = value; return S1; }
			@Override public State handleDot(FloatData data){return S2;}
		}, 
		S1() {
			@Override public State handleDigit(FloatData data, int value) { data.m = 10*data.m + value; return S1; }
			@Override public State handleDot(FloatData data) { return S3; }
			@Override public State handleE(FloatData data) { return S4; }
		}, 
		S2() {
			@Override public State handleDigit(FloatData data, int value) { data.m = data.m + value/data.quo; data.quo = 10*data.quo; return S3; }
		}, 
		S3() {
			@Override public State handleDigit(FloatData data, int value) { data.m = data.m + value/data.quo; data.quo = 10*data.quo; return S3; }
			@Override public State handleE(FloatData data){return S4;}
		}, 
		S4() {
			@Override public State handlePlus(FloatData data) { return S5; }
			@Override public State handleMinus(FloatData data) { data.exp_sign = -1; return S5; }
			@Override public State handleDigit(FloatData data, int value) { data.exp = value; return S6; }
		}, 
		S5() {
			@Override public State handleDigit(FloatData data, int value) { data.exp = value; return S6; }
		}, 
		S6() {
			@Override public State handleDigit(FloatData data, int value) { data.exp = 10*data.exp + value; return S6; }
		},
		ERROR;

		public State handleDigit(FloatData data, int value) { return ERROR; }
		public State handleE(FloatData data) { return ERROR; }
		public State handleDot(FloatData data) { return ERROR; }
		public State handlePlus(FloatData data) { return ERROR; }
		public State handleMinus(FloatData data) { return ERROR; }

	}

	static class FloatData {
		double m = 0, quo=10;
		int exp = 0, exp_sign = 1;
		double getValue() { return m * Math.pow(10, exp_sign * exp); }
	}

	private static double parseFloat(String str) {
		State s = State.S0;
		FloatData data = new FloatData();
		int pos = 0;
		while(s != State.ERROR && pos < str.length()) {
			char ch = str.charAt(pos++);
			if(Character.isDigit(ch)) s = s.handleDigit(data, Character.getNumericValue(ch));
			else if(ch == '.') s = s.handleDot(data);
			else if(ch == '+') s = s.handlePlus(data);
			else if(ch == '-') s = s.handleMinus(data);
			else if(ch == 'E') s = s.handleE(data);
			else if(ch == 'e') s = s.handleE(data);
			else s = State.ERROR;
		}		
		if(s == State.S3 || s == State.S6) {
			return data.getValue();
		}
		else {
			throw new IllegalArgumentException();
		}
	}
}
