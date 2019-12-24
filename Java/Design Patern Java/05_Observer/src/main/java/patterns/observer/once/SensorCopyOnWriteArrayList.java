package patterns.observer.once;

import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;

public class SensorCopyOnWriteArrayList extends Sensor {
	private final List<Observer> observables = new CopyOnWriteArrayList<>();

	@Override
	public void addObserver(Observer o) {
		observables.add(o);
	}

	@Override
	public void removeObserver(Observer o) {
		observables.remove(o);
	}

	@Override
	public void notifyObservers() {
		for(Observer o : observables) {
			o.update(this);
		}
	}

}
