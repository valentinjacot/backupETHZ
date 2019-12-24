package patterns.observer.cycle;

import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;

public class Observable {
	private List<Observer> observers = new CopyOnWriteArrayList<>();

	public void addObserver(Observer o) {
		observers.add(o);
	}

	public void removeObserver(Observer o) {
		observers.remove(o);
	}

	protected void notifyObservers(Object arg) {
		for (Observer obs : observers) {
			obs.update(this, arg);
		}
	}
}
