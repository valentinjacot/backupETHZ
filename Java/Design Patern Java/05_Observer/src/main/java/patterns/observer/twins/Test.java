package patterns.observer.twins;

import java.util.Observer;

@SuppressWarnings("deprecation")
public class Test {

	public static void main(String[] args) {
		ObservableModel model = new ObservableModel();
		Observer obs = new PrintObserver();
		model.addObserver(obs);
		model.setValue(5);
		model.setValue(10);
	}

}
