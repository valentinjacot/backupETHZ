	package patterns.observer.twins;

import java.util.Observable;
import java.util.Observer;

@SuppressWarnings("deprecation")
public class PrintObserver implements Observer {
	@SuppressWarnings("unchecked")
	public void update(Observable obs, Object arg) {
		Model m = ((ObserverHelper<Model>) obs).getObservableModel();
		System.out.println("new value: " + m.getValue());
	}
}
