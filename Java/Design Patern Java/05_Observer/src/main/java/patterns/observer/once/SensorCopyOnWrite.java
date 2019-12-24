package patterns.observer.once;

import java.util.ArrayList;
import java.util.List;

public class SensorCopyOnWrite extends Sensor {

	private List<Observer> observables = new ArrayList<>();

	@Override
	public void addObserver(Observer o) {
		List<Observer> tmp = new ArrayList<>(observables);
		tmp.add(o);
		observables = tmp;
	}

	@Override
	public void removeObserver(Observer o) {
		List<Observer> tmp = new ArrayList<>(observables);
		tmp.remove(o);
		observables = tmp;
	}

	@Override
	public void notifyObservers() {
		for (Observer o : observables) {
			o.update(this);
		}
	}
}
