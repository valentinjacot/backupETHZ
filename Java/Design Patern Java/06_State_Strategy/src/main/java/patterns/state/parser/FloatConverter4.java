package patterns.state.parser;

import java.io.BufferedReader;
import java.io.InputStreamReader;

public class FloatConverter4 {
	
	public static void main(String[] args) throws Exception {
		BufferedReader r = new BufferedReader(new InputStreamReader(System.in));
		String s = r.readLine();
		while(s != null && s.length() > 0){
			try {double d = parseFloat(s); System.out.println(d);}
			catch(IllegalArgumentException e){System.out.println("Illegal Format");}
			s = r.readLine();
		}
	}

	private static interface Context {
		void setState(State s);
		FloatData getData();
	}

	private static class FloatData {
		double m = 0, quo=10;
		int exp = 0, exp_sign = 1;
		double getValue(){return m * Math.pow(10, exp_sign * exp);}
	}


	private static interface State {
		void handleDigit(int val);
		void handleE();
		void handleDot();
		void handlePlus();
		void handleMinus();
	}
	
	private static abstract class AbstractState implements State {
		protected final Context c;
		protected AbstractState(Context c) {
			this.c = c;
		}
		@Override public void handleDigit(int value) { c.setState(ERROR); }
		@Override public void handleE() { c.setState(ERROR); }
		@Override public void handleDot() { c.setState(ERROR); }
		@Override public void handlePlus() { c.setState(ERROR); }
		@Override public void handleMinus() { c.setState(ERROR); }
	}
	
	private static class S0 extends AbstractState {
		S0(Context c) { super(c); }
		@Override public void handleDigit(int value) { c.getData().m = value; c.setState(S1);  }
		@Override public void handleDot() { c.setState(S2); }
	}
	
	private static class S1 extends AbstractState {
		S1(Context c) { super(c); }
		@Override public void handleDigit(int value) { FloatData data = c.getData(); data.m = 10*data.m + value;}
		@Override public void handleDot() { c.setState(S3);}
		@Override public void handleE() { c.setState(S4);}
	}
	
	private static class S2 extends AbstractState {
		S2(Context c) { super(c); }
		@Override public void handleDigit(int value) { FloatData data = c.getData(); data.m = data.m + value/data.quo; data.quo = 10*data.quo; c.setState(S3);}
	}
	
	private static class S3 extends AbstractState {
		S3(Context c) { super(c); }
		@Override public void handleDigit(int value) { FloatData data = c.getData(); data.m = data.m + value/data.quo; data.quo = 10*data.quo; }
		@Override public void handleE() { c.setState(S4);}
	}
	
	private static class S4 extends AbstractState {
		S4(Context c) { super(c); }
		@Override public void handlePlus() { c.setState(S5);}
		@Override public void handleMinus() { FloatData data = c.getData(); data.exp_sign = -1; c.setState(S5); }
		@Override public void handleDigit(int value) { FloatData data = c.getData(); data.exp = value; c.setState(S6); }
	}
	
	private static class S5 extends AbstractState {
		S5(Context c) { super(c); }
		@Override public void handleDigit(int value) { FloatData data = c.getData(); data.exp = value; c.setState(S6); }
	}
	
	private static class S6 extends AbstractState {
		S6(Context c) { super(c); }
		@Override public void handleDigit(int value) { FloatData data = c.getData(); data.exp = 10*data.exp + value; }
	}
	
	private static class ErrorState extends AbstractState {
		ErrorState(Context c) { super(c); }
	}

	private static final FloatData data = new FloatData();
	
	private static final Context c = new Context() {
		@Override
		public void setState(State s) {
			FloatConverter4.s = s;
		}
		@Override
		public FloatData getData() {
			return data;
		}	
	};
	
	private static final State ERROR = new ErrorState(c);
	private static final State S0 = new S0(c);
	private static final State S1 = new S1(c);
	private static final State S2 = new S2(c);
	private static final State S3 = new S3(c);
	private static final State S4 = new S4(c);
	private static final State S5 = new S5(c);
	private static final State S6 = new S6(c);

	private static State s;
	
	private static double parseFloat(String str) {
		s = S0;
		int pos = 0;
		while(s != ERROR && pos < str.length()) {
			char ch = str.charAt(pos++);
			if(Character.isDigit(ch)) s.handleDigit(Character.getNumericValue(ch));
			else if(ch == '.') s.handleDot();
			else if(ch == '+') s.handlePlus();
			else if(ch == '-') s.handleMinus();
			else if(ch == 'E') s.handleE();
			else if(ch == 'e') s.handleE();
			else c.setState(ERROR);
		}		
		if(s == S3 || s == S6) {
			return data.getValue();
		}
		else {
			throw new IllegalArgumentException();
		}
	}
}
