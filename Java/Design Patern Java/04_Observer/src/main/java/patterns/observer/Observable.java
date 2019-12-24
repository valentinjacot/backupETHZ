package patterns.observer;

import java.util.ArrayList;
import java.util.List;

public class Observable {
	private final List<Observer> observers = new ArrayList<>();

	public void addObserver(Observer o) {
		observers.add(o);
	}

	public void removeObserver(Observer o) {
		observers.remove(o);
	}

	protected void notifyObservers() {
		for (Observer obs : observers) {
			obs.update(this);
		}
		// Variant to the for loop
		// observers.forEach(obs -> obs.update(this));
	}
}
