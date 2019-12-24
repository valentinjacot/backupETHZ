package patterns.observer.once;

public class ObserverTest {

	public static void main(String[] args) {
		Sensor s = new SensorConcurrentModificationException();
		// Sensor s = new SensorIterateOverCopy();
		// Sensor s = new SensorPendingMutations();
		// Sensor s = new SensorCopyOnWrite();
		// Sensor s = new SensorCopyOnWriteArrayList();

		Observer po1 = new PrintObserver("Printer 1");
		Observer po2 = new PrintObserver("Printer 2");
		Observer po3 = new PrintObserver("Printer 3");
		Observer oo = new OnceObserver();

		// s.addObserver(new RemoveAllObserver(new Observer[]{po1,po2,po3}));
		s.addObserver(po1);
		s.addObserver(oo);
		s.addObserver(po2);
		s.addObserver(po3);

		s.setValue(22);
		System.out.println();
		s.setValue(30);
	}

}
