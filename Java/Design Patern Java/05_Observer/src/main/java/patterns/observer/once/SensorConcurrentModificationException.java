package patterns.observer.once;

import java.util.ArrayList;
import java.util.List;

public class SensorConcurrentModificationException extends Sensor {
	private List<Observer> observers = new ArrayList<Observer>();

	@Override
	public void addObserver(Observer o) {
		observers.add(o);
	}

	@Override
	public void removeObserver(Observer o) {
		observers.remove(o);
	}

	@Override
	public void notifyObservers() {
		for (Observer obs : observers) {
			obs.update(this);
		}
//		Iterator<Observer> it = observers.iterator();
//		while(it.hasNext()) {
//			it.next().update(this);
//		}
	}
}
