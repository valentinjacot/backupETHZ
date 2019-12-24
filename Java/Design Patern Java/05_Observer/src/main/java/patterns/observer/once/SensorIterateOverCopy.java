package patterns.observer.once;

import java.util.ArrayList;
import java.util.List;

public class SensorIterateOverCopy extends Sensor {

	private final List<Observer> observables = new ArrayList<>();

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
		for(Observer o : new ArrayList<>(observables)) {
			o.update(this);
		}
	}
}
