package patterns.observer.once;

public class RemoveAllObserver implements Observer {
	
	private Observer[] observers;

	public RemoveAllObserver(Observer[] observers) {
		this.observers = observers;
	}

	@Override
	public void update(Observable bag) {
		System.out.println("Remove All Observer called");
		for (Observer obs : observers) {
			bag.removeObserver(obs);
		}
	}
}
