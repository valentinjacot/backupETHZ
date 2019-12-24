package patterns.observer.once;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

public class SensorPendingMutations extends Sensor {

	private final List<Observer> observers = new ArrayList<>();
	private final List<Mutation> pendingMutations = new LinkedList<>();
	private int level = 0;

	@Override
	public final void addObserver(Observer o) {
		if (level > 0) {
			pendingMutations.add(new Mutation(o, true));
		} else {
			observers.add(o);
		}
	}

	@Override
	public final void removeObserver(Observer o) {
		if (level > 0) {
			pendingMutations.add(new Mutation(o, false));
		} else {
			observers.remove(o);
		}
	}

	@Override
	public final void notifyObservers() {
		level++;
		for (Observer obs : observers)
			obs.update(this);
		level--;
		if (level == 0) {
			for (Mutation m : pendingMutations) {
				if (m.add) observers.add(m.observer);
				else observers.remove(m.observer);
			}
			pendingMutations.clear();
		}
	}

	private static class Mutation {
		private final Observer observer;
		private final boolean add;

		public Mutation(Observer observer, boolean add) {
			this.observer = observer;
			this.add = add;
		}
	}
}
